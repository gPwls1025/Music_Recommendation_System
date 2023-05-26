import os
import sys
import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
import pyspark
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics
import csv 
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType

def main(spark, userID, file_size):
    
    # Load the track, train, val from Parquet files
    if file_size == 'small':

        train = spark.read.parquet(f'hdfs:/user/{userID}/recommender_train_small')
        val = spark.read.parquet(f'hdfs:/user/{userID}/recommender_val_small')
    
    elif file_size == 'full':

        train = spark.read.parquet(f'hdfs:/user/{userID}/recommender_train')
        val = spark.read.parquet(f'hdfs:/user/{userID}/recommender_val')

    # Create SQL tables
    train.createOrReplaceTempView("train")
    val.createOrReplaceTempView("val")
    
    # Load test data and test data preparation
    track_test = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet', header=True)
    track_test.printSchema()
    track_test.createOrReplaceTempView('track_test')
    
    interactions_test = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet', header=True)
    interactions_test.printSchema()
    interactions_test.createOrReplaceTempView('interactions_test') 

    joined_test_df = track_test.join(interactions_test, on="recording_msid", how="inner")
    
    print("start converting to RDD")
    # Convert the train, validation set dataframe to an RDD of (user, item) tuples
    val_rdd = val.select("user_id", "recording_msid").rdd
    train_rdd = train.select("user_id", "recording_msid").rdd
    test_rdd = joined_test_df.select("user_id", "recording_msid").rdd

    # Group the train, validation data by user_id and collect the recording_msid as a list
    val_grouped = val_rdd.groupByKey().mapValues(list)
    train_grouped = train_rdd.groupByKey().mapValues(list)
    test_grouped = test_rdd.groupByKey().mapValues(list)
    

    # Factor in total interaction count and distinct user count
    # Calculate the total count of interactions for each item
    item_interactions = train.groupBy('recording_msid').agg(func.sum('count').alias('total_interactions'))

    # Calculate the distinct user count for each item
    distinct_users = train.groupBy('recording_msid').agg(func.count('user_id').alias('distinct_users'))

    # Join the two dataframes on recording_msid
    popularity_scores = item_interactions.join(distinct_users, on='recording_msid')

    C = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # C = [0, .1]
    results = [] # List to store the results

    
    print("start calculating MAP")
    for c in C:
        print(f'start iteration: {c}\n')
        # Calculate the popularity score, weighting more on distinct user count
        popularity_scores = popularity_scores.withColumn('popularity_score', 
                                                        func.round((c*func.log(func.col('distinct_users') + 1) + (1-c)*func.log(func.col('total_interactions') + 1)), 2)) 

        # Sort the items by popularity score and select the top 100
        popular_100 = popularity_scores.orderBy('popularity_score', ascending=False).limit(100)

        # Create an RDD of top 100 recommended items
        popular_100_rdd = popular_100.rdd.map(lambda x: x[0]).collect()

        # Create a list of (user's recording_msid, top100_recommended_items) tuples for each user in validation and train
        val_pred_and_actual = val_grouped.map(lambda x: (x[1], popular_100_rdd))
        train_pred_and_actual = train_grouped.map(lambda x: (x[1], popular_100_rdd))
        test_pred_and_actual = test_grouped.map(lambda x: (x[1], popular_100_rdd))

        # Instantiate a RankingMetrics object and pass in the val_pred_and_actual and train_pred_and_actual RDD
        val_metrics = RankingMetrics(val_pred_and_actual)
        train_metrics = RankingMetrics(train_pred_and_actual)
        test_metrics = RankingMetrics(test_pred_and_actual)
        
        print(f'finish iteration: {c}\n')

        # Append the results to the list
        results.append((str(c), val_metrics.meanAveragePrecision, val_metrics.meanAveragePrecisionAt(100), val_metrics.precisionAt(100), val_metrics.recallAt(100), 
                        train_metrics.meanAveragePrecision, train_metrics.meanAveragePrecisionAt(100), train_metrics.precisionAt(100), train_metrics.recallAt(100), 
                        test_metrics.meanAveragePrecision, test_metrics.meanAveragePrecisionAt(100), test_metrics.precisionAt(100), test_metrics.recallAt(100)))
        
    print("start writing the spark-df and csv")
    # columns = ['parameter', 'val_mean_avg_precision', 'val_mean_avg_precision_100', 'val_avg_precision_100', 'val_avg_recall_100',
    #                 'train_mean_avg_precision', 'train_mean_avg_precision_100', 'train_avg_precision_100', 'train_avg_recall_100',
    #                 'test_mean_avg_precision', 'test_mean_avg_precision_100', 'test_avg_precision_100', 'test_avg_recall_100']
    # df = pd.DataFrame(results, columns=columns)
    # df.to_csv(f'hdfs:/user/{userID}/results.csv', index=False)
    # df.write.format("csv").option("header", "true").mode("overwrite").save(f'hdfs:/user/{userID}/results.csv')

    # Define the schema of the DataFrame
    schema = StructType([StructField("parameter", StringType(), True),
                         StructField("val_mean_avg_precision", FloatType(), True),
                         StructField("val_mean_avg_precision_100", FloatType(), True),
                         StructField("val_avg_precision_100", FloatType(), True),
                         StructField("val_avg_recall_100", FloatType(), True),
                         StructField("train_mean_avg_precision", FloatType(), True),
                         StructField("train_mean_avg_precision_100", FloatType(), True),
                         StructField("train_avg_precision_100", FloatType(), True),
                         StructField("train_avg_recall_100", FloatType(), True),
                         StructField("test_mean_avg_precision", FloatType(), True),
                         StructField("test_mean_avg_precision_100", FloatType(), True),
                         StructField("test_avg_precision_100", FloatType(), True),
                         StructField("test_avg_recall_100", FloatType(), True),
                         ])
    
    # Create a PySpark DataFrame from the list of tuples and the schema
    df = spark.createDataFrame(results, schema=schema)

    if file_size == 'small':
        df.write.format("csv").option("header", "true").mode("overwrite").save(f'hdfs:/user/{userID}/results_{file_size}.csv')

    elif file_size == 'full':
        df.write.format("csv").option("header", "true").mode("overwrite").save(f'hdfs:/user/{userID}/results_{file_size}.csv')
    
    print('finish')


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)
