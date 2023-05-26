import pyspark
import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr
from scipy.sparse import csr_matrix
import numpy as np
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.window import Window

def main(spark, userID, file_size):
    #------------------------Load the track, train, val from Parquet files------------------------
    print('\nLoading data...\n')

    if file_size == 'small':
        train = spark.read.parquet(f'hdfs:/user/{userID}/recommender_train_small')
        val = spark.read.parquet(f'hdfs:/user/{userID}/recommender_val_small')
    elif file_size == 'full':
        train = spark.read.parquet(f'hdfs:/user/{userID}/recommender_train')
        val = spark.read.parquet(f'hdfs:/user/{userID}/recommender_val')

    # Load test data
    test_filtered = spark.read.parquet(f'hdfs:/user/{userID}/recommender_test')

    # Create SQL tables
    train.createOrReplaceTempView("train")
    val.createOrReplaceTempView("val")
    test_filtered.createOrReplaceTempView('test_filtered')

    #------------------------Filter the data------------------------
    print("Start filtering data...")
    # Add flag variable for train/test set
    train = train.withColumn('has_history', func.lit(1))
    test_filtered = test_filtered.withColumn('is_train', func.lit(0))

    # Find the users that have no interaction history in the merged DataFrame
    users_without_history = test_filtered.select('user_id').distinct().subtract(train.select('user_id').distinct())
    # Find the cold start items for each user in the test set
    cold_start_items = test_filtered.join(users_without_history, on='user_id', how='leftsemi')
    cold_start_items.createOrReplaceTempView('cold_start_items')

    # Create a new column called "has_history" in the test_filtered DataFrame, if user_id in cold_start_items matches with user_id in test_filtered dataset then 0, otherwise 1 
    #test_filtered = test_filtered.withColumn('has_history', func.when(func.col('user_id'), func.col('user_id').isin(cold_start_items.collect()), 0).otherwise(1))
    test_temp = spark.sql(("""SELECT test_filtered.*, CASE WHEN cold_start_items.user_id IS NULL THEN 1 ELSE 0 END AS has_history FROM test_filtered LEFT JOIN cold_start_items ON test_filtered.user_id = cold_start_items.user_id"""))
    test_temp.show()
    # Count the number of rows in test_temp where has_history = 1
    num_has_history = test_temp.filter(test_temp.has_history == 1).count()
    num_no_history = test_temp.filter(test_temp.has_history == 0).count()
    print("has history in test set:", num_has_history)
    print("no history in test set:", num_no_history)

    #---------------------Nearest Neighbor---------------------
    k=100
    
    print("Start nearest neighbor")
    #drop has_history column in train since val doesn't have has_history column
    train = train.drop('has_history')
    
    # Index user_id and recording_msid
    indexer_user = StringIndexer(inputCol='user_id', outputCol='user_id_index')
    indexer_recording = StringIndexer(inputCol='recording_msid', outputCol='recording_msid_index')

    #transform data into feature vectors
    assembler = VectorAssembler(inputCols=["user_id_index", "recording_msid_index"], outputCol="features")

    # Define the LSH model
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=3.0, numHashTables=3)

    # Define the pipeline
    pipeline = Pipeline(stages=[indexer_user, indexer_recording, assembler, brp])

    # Fit the model on the training data
    model = pipeline.fit(train)

    # Transform the data
    train_transformed = model.transform(train)
    val_transformed = model.transform(val)
    
    key = Vectors.dense(1.0, 0.0)
    
    # Do the Approximate Nearest Neighbor Search
    result = model.stages[-1].approxNearestNeighbors(train_transformed, key, 500)
    result.show()
    result.printSchema()

    #-------------------Recommmend songs-----------------
    # Group the result table by user_id and sort each group by distCol
    windowSpec = Window.partitionBy("user_id").orderBy("distCol")

    # Add a rank column to each group
    ranked_df = result.withColumn("rank", func.row_number().over(windowSpec))

    # Filter the ranked_df to keep only the top k records for each user
    rec_df = ranked_df.filter(col("rank") <= k)

    # Show the recommended songs for each user
    per_rec = rec_df.groupBy("user_id").agg(func.collect_list("recording_msid").alias("recommended_songs"))
    per_rec.show(truncate=False)

   #-------------------Evaluate-----------------
    #filter our test set to only include users with history
    test = test_temp.filter(test_temp.has_history == 1)
    
    # Convert to RDD
    predictions_rdd = per_rec.select('user_id', 'recommended_songs') \
        .rdd \
        .map(lambda row: (row[0], row[1]))

    # Prepare ground truth
    ground_truth = val.groupBy("user_id").agg(func.collect_list("recording_msid").alias("ground_truth"))
    #ground_truth = test.groupBy("user_id").agg(func.collect_list("recording_msid").alias("ground_truth"))

    # Convert to RDD
    ground_truth_rdd = ground_truth.select('user_id', 'ground_truth') \
        .rdd \
        .map(lambda row: (row[0], row[1]))

    # Join the prediction RDD and ground truth RDD
    joined_rdd = predictions_rdd.join(ground_truth_rdd) \
        .map(lambda row: (row[1][0], row[1][1]))

    # Create a RankingMetrics instance
    metrics = RankingMetrics(joined_rdd)

    # Compute Mean Average Precision
    map_score = metrics.meanAveragePrecision
    map_score100 = metrics.meanAveragePrecisionAt(k)
    precision100 = metrics.precisionAt(k)
    recall100 = metrics.recallAt(k)
    print(f"MAP = {map_score}\n")
    print(f"MAP@100 = {map_score100}\n")
    print(f"Precision@100 = {precision100}\n")
    print(f"Recall@100 = {recall100}\n")
    print('finish')

if __name__ == "__main__":
        
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)
