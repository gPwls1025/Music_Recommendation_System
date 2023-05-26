import pyspark
import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr

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

    #------------------------Since ALS takes integer index, convert recording_msid to index------------------------
    
    print('\nIndexing...\n')
    
    # Initialize the StringIndexer
    msid_indexer = StringIndexer(inputCol="recording_msid", outputCol="recording_index", handleInvalid="keep")

    # Fit the indexer to training data
    msid_model = msid_indexer.fit(train)

    # Transform training data
    indexed_train = msid_model.transform(train)

    # # Transform val data using the same 'msid_model' fitted on the training data
    # indexed_val = msid_model.transform(val)

    # Transform test data using the same 'msid_model' fitted on the training data
    indexed_test = msid_model.transform(test_filtered)

    #------------------------Begin ALS------------------------
    
    print('\nBegin ALS...\n')
    
    # Initialize the ALS model
    als = ALS(
        rank=200,
        maxIter=15,
        regParam=1,
        alpha=10,
        userCol="user_id",
        itemCol="recording_index",
        ratingCol="count",
        coldStartStrategy="drop",
        implicitPrefs=True,
        nonnegative=True
    )

    # Fit the ALS model to the indexed training data
    als_model = als.fit(indexed_train)
    
    # Get the unique users from the test set
    user_subset = indexed_test.select("user_id").distinct()

    K = 100

    # Generate top-K recommendations for the user subset
    user_rec = als_model.recommendForUserSubset(user_subset, K)

    # Prepare the val set in a suitable format for RankingMetrics
    indexed_test = indexed_test.select(col("user_id"), col("recording_index").alias("label"))
    indexed_test = indexed_test.groupBy("user_id").agg(expr("collect_list(label) as labels"))

    # Prepare the recommendations in a suitable format for RankingMetrics
    user_rec = user_rec.select(col("user_id"), col("recommendations.recording_index").alias("pred"))
    user_rec = user_rec.join(indexed_test, "user_id", "inner")

    # Convert predictions and labels columns to RDD format
    user_rec_rdd = user_rec.select("pred", "labels").rdd

    # Instantiate a RankingMetrics object with the prepared data
    metrics = RankingMetrics(user_rec_rdd)
    
    #------------------------Evaluation------------------------

    print('\nBegin evaluation...\n')
    
    # Compute the MAP score
    map_score = metrics.meanAveragePrecision
    map_score100 = metrics.meanAveragePrecisionAt(K)
    precision100 = metrics.precisionAt(K) 
    recall100 = metrics.recallAt(K)

    print(f"MAP = {map_score}\n")
    print(f"MAP@100 = {map_score100}\n")
    print(f"Precision@100 = {precision100}\n")
    print(f"Recall@100 = {recall100}\n")
    print('finish')

    # Open a text file for writing
    with open('output.txt', 'w') as f:
        # Write the MAP score to the file
        f.write(f"MAP = {map_score}\n")
        # Write the MAP@100 score to the file
        f.write(f"MAP@100 = {map_score100}\n")
        # Write the Precision@100 score to the file
        f.write(f"Precision@100 = {precision100}\n")
        # Write the Recall@100 score to the file
        f.write(f"Recall@100 = {recall100}\n")
        # Write 'finish' to the file to indicate completion
        f.write('finish\n')

    # Print a message to indicate completion
    print('\nEvaluation complete.\n')


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)
