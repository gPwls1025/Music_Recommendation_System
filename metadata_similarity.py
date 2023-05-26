import pyspark
import os
import sys
import numpy as np
import pandas as pd
import annoy 
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler, Normalizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex

def main(spark, userID, file_size):
    #------------------------Load the track, train, val from Parquet files------------------------
    print('\nLoading data...\n')

    if file_size == 'small':
        train = spark.read.parquet(f'hdfs:/user/{userID}/extension_train_small')
        val = spark.read.parquet(f'hdfs:/user/{userID}/extension_val_small')
    elif file_size == 'full':
        train = spark.read.parquet(f'hdfs:/user/{userID}/recommender_train')
        val = spark.read.parquet(f'hdfs:/user/{userID}/recommender_val')

    # Load test data
    test_filtered = spark.read.parquet(f'hdfs:/user/{userID}/extension_test')
    
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
    # test_temp.show()
    # Count the number of rows in test_temp where has_history = 1
    num_has_history = test_temp.filter(test_temp.has_history == 1).count()
    num_no_history = test_temp.filter(test_temp.has_history == 0).count()
    print("has history in test set:", num_has_history)
    print("no history in test set:", num_no_history)

    #---------------------Metadata similarity---------------------
    print("Start the progress for metadata similarity")
    #drop has_history column in train since val doesn't have has_history column
    train = train.drop('has_history')
    
    #filter our test set to only include users without history
    test = test_temp.filter(test_temp.has_history == 0)
    test.show() 
    test.columns
    # Define metadata columns
    metadata_columns = ["artist_name", "track_name", "season", "count"]
    
    # Preprocess the data (train, val, test)
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="skip") for col in metadata_columns]
    indexers_pipeline = Pipeline(stages=indexers)
    train_indexed = indexers_pipeline.fit(train).transform(train)
    validation_indexed = indexers_pipeline.fit(val).transform(val)
    test_indexed = indexers_pipeline.fit(test).transform(test)

    # Assemble the metadata features into a single vector
    assembler = VectorAssembler(inputCols=[col+"_index" for col in metadata_columns], outputCol="features")
    train_features = assembler.transform(train_indexed)
    validation_features = assembler.transform(validation_indexed)
    test_features = assembler.transform(test_indexed)

    # Normalize the feature vectors
    normalizer = Normalizer(inputCol="features", outputCol="normalized_features")
    train_normalized = normalizer.transform(train_features)
    validation_normalized = normalizer.transform(validation_features)
    test_normalized = normalizer.transform(test_features)

    # Extract the metadata features from train_normalized and validation_normalized
    train_metadata = train_normalized.select("recording_msid", "normalized_features").rdd.map(lambda x: (x[0], Vectors.dense(x[1])))
    validation_metadata = validation_normalized.select("recording_msid", "normalized_features").rdd.map(lambda x: (x[0], Vectors.dense(x[1])))
    test_metadata = test_normalized.select("recording_msid", "normalized_features").rdd.map(lambda x: (x[0], Vectors.dense(x[1])))

    print("Start the progress for computing cosine similarity matrix")
    # Compute cosine similarity matrix
    #The cosine_similarity_matrix will be a NumPy array representing the pairwise cosine similarity scores between the two sets.
    cosine_similarity_matrix = cosine_similarity(np.array(test_metadata.map(lambda x: x[1]).collect()), np.array(train_metadata.map(lambda x: x[1]).collect()))
    
    # Build the annoy index using normalized features of the training set tracks:
    # Define the number of dimensions in the feature vectors
    num_dimensions = len(metadata_columns)
    # Build the annoy index
    annoy_index = AnnoyIndex(num_dimensions, metric="angular")  # Use angular distance for cosine similarity
    # Add items to the annoy index
    for i, (_, track_id) in enumerate(train_metadata):
        annoy_index.add_item(i, track_id)
    # Build the annoy index
    annoy_index.build(n_trees=10)  # Adjust the number of trees as needed
    
    print("Generate recommendations for test set tracks using the annoy index")
    # Generate recommendations for test set tracks using the annoy index
    top_k = 100
    recommendations_test = []
    for test_track_id, _ in test_metadata:
        neighbors = annoy_index.get_nns_by_vector(test_track_id, top_k, include_distances=False)
        recommendations_test.append((test_track_id, neighbors))

    # Convert the recommendations to DataFrame
    recommend_tracks_test = spark.createDataFrame(recommendations_test, ["track_id", "recommendations"])

    # Group the recommendations by user and collect the track IDs
    recommendations_by_user =recommend_tracks_test.groupBy("user_id").agg(F.collect_list("recommendations").alias("recommendations"))

    # Show the recommendations for each user
    recommendations_by_user.show()

    #-------------------Evaluation-----------------
    print("Start the progress for evaluation") 
    

    
if __name__ == "__main__":
        
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)




