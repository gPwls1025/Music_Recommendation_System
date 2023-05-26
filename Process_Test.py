import pyspark
import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr

def main(spark, userID):
    
    #------------------------Change test set to a suitable format------------------------
    
    # Load test data and test data preparation
    track_test = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet', header=True)
    track_test.printSchema()
    track_test.createOrReplaceTempView('track_test')
    
    interactions_test = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet', header=True)
    interactions_test.printSchema()
    interactions_test.createOrReplaceTempView('interactions_test') 

    print('\nConvert test set...\n')
    
    # Join track and interaction table on recording_msid
    joined_df = track_test.join(interactions_test, on="recording_msid", how="inner")

    # If join recording_msid has mbid, replace all of that user's specific msid with mbid
    joined_df = joined_df.withColumn("recording_msid", func.when(joined_df["recording_mbid"].isNotNull(), joined_df["recording_mbid"]).otherwise(joined_df["recording_msid"]))

    # For each user_id, get the count of unique msid.
    joined_df = joined_df.groupBy("user_id", "recording_msid") \
                             .agg(func.count("recording_msid") \
                             .alias("count")) \
                             .orderBy("user_id", "count", ascending=True)
    
    # Filter out users with less than 10 unique recording_msid. 
    joined_userid_filter = joined_df.groupBy("user_id") \
                                    .agg(func.countDistinct("recording_msid") \
                                    .alias("count")) \

    joined_userid_filter = joined_userid_filter.filter(joined_userid_filter["count"] >= 5)
    joined_filtered = joined_df.join(joined_userid_filter, "user_id")

    # Filter out recording_msid with less than 10 unique user_id
    joined_msid_filter = joined_filtered.groupBy("recording_msid") \
                                    .agg(func.countDistinct("user_id") \
                                    .alias("count")) \

    joined_msid_filter = joined_msid_filter.filter(joined_msid_filter["count"] >= 5)
    joined_filtered = joined_df.join(joined_msid_filter, "recording_msid")

    # For each user_id, get the count of unique msid.
    joined_filtered = joined_filtered.groupBy("user_id", "recording_msid") \
                                    .agg(func.count("recording_msid") \
                                    .alias("count")) \
                                    .orderBy("user_id", "count", ascending=True)
    
    # Save test
    joined_filtered.select('user_id', 'recording_msid', 'count').write.parquet(f'hdfs:/user/{userID}/recommender_test')


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS').config("spark.sql.broadcastTimeout", "36000").getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    # Call our main routine
    main(spark,userID)
