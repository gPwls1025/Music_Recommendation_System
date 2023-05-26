import os
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as func 
import pyspark
from pyspark.sql.window import Window
from pyspark.sql.functions import countDistinct

# spark-submit --deploy-mode client lab_3_starter_code.py 

# If you are running this script on the cluster, you will need to use the following command:
# spark-submit --deploy-mode cluster --py-files bench.py basic_query.py <your_data_file_path>
# Cluster mode is good if my local cluster is running out of memory. It launches on a different machine.
# 

def main(spark, userID, file_size):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    # -------------------Load Data---------------------
    
    print('\nLoading the files from HDFS...\n')
    
    # Load the files
    if file_size == 'small':

        track = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet', header=True)
        track.printSchema()
        track.createOrReplaceTempView('track')
        
        interaction = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet', header=True)
        interaction.printSchema()
        interaction.createOrReplaceTempView('interaction') 

    elif file_size == 'full':

        track = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet', header=True)
        track.printSchema()
        track.createOrReplaceTempView('track')
        
        interaction = spark.read.parquet('/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet', header=True)
        interaction.printSchema()
        interaction.createOrReplaceTempView('interaction') 


    # -------------------Filter Data---------------------

    print('\nFiltering the data...\n')

    # Join track and interaction table on recording_msid
    joined_df = track.join(interaction, on="recording_msid", how="inner")

    # If join recording_msid has mbid, replace all of that user's specific msid with mbid
    joined_df = joined_df.withColumn("recording_msid", func.when(joined_df["recording_mbid"].isNotNull(), joined_df["recording_mbid"]).otherwise(track["recording_msid"]))

    # Filter out users with less than 10 unique recording_msid. 
    joined_userid_filter = joined_df.groupBy("user_id") \
                                    .agg(func.countDistinct("recording_msid") \
                                    .alias("count")) \

    joined_userid_filter = joined_userid_filter.filter(joined_userid_filter["count"] >= 10)
    joined_filtered = joined_df.join(joined_userid_filter, "user_id")

    # Filter out recording_msid with less than 10 unique user_id
    joined_msid_filter = joined_filtered.groupBy("recording_msid") \
                                    .agg(func.countDistinct("user_id") \
                                    .alias("count")) \

    joined_msid_filter = joined_msid_filter.filter(joined_msid_filter["count"] >= 10)
    joined_filtered = joined_df.join(joined_msid_filter, "recording_msid")

    # For each user_id, get the count of unique msid.
    joined_filtered = joined_filtered.groupBy("user_id", "recording_msid") \
                                    .agg(func.count("recording_msid") \
                                    .alias("count")) \
                                    .orderBy("user_id", "count", ascending=True)
    
    # -------------------Split Data---------------------
    
    print('\nSplitting the data...\n')

    # Create a window partitioned by user_id and ordered by a random number
    user_window = Window.partitionBy("user_id").orderBy(func.rand())

    # Assign a row number to each interaction within each user's partition
    joined_filtered = joined_filtered.withColumn("row_num", func.row_number().over(user_window))

    # Calculate the total number of interactions for each user_id
    user_counts = joined_filtered.groupBy("user_id").agg(func.count("*").alias("total_count"))

    # Join the total count to the original DataFrame
    joined_filtered = joined_filtered.join(user_counts, "user_id")

    # Assign interactions to train or validation set based on row number
    val = joined_filtered.filter(func.col('row_num') > (0.8 * joined_filtered["total_count"]))
    train = joined_filtered.filter(func.col('row_num') <= (0.8 * joined_filtered["total_count"]))
    
    # Leave 20% as a hold out data (by assigning zero) so we can compare the predction with the validation set later 
    # train = joined_filtered.withColumn("count", func.when(func.col('row_num') > (0.8 * joined_filtered["total_count"]), 0) \
    #                                 .otherwise(joined_filtered["count"]))
    
    # Drop unnecessary columns
    val = val.drop("row_num", "total_count")
    train = train.drop("row_num", "total_count")

    # -------------------Save Data---------------------
    print('\nSaving the data...\n')

    #Save the results to parquet
    if file_size == 'small':

        val.select('user_id', 'recording_msid', 'count', 'artist_name', 'track_name').write.parquet(f'hdfs:/user/{userID}/recommender_val_small')
        train.select('user_id', 'recording_msid', 'count', 'artist_name', 'track_name').write.parquet(f'hdfs:/user/{userID}/recommender_train_small')

    elif file_size == 'full':

        val.select('user_id', 'recording_msid', 'count', 'artist_name', 'track_name').write.parquet(f'hdfs:/user/{userID}/recommender_val')
        train.select('user_id', 'recording_msid', 'count', 'artist_name', 'track_name').write.parquet(f'hdfs:/user/{userID}/recommender_train')
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('read_files').getOrCreate()

    userID = os.environ['USER']

    # User can specify the file size: small or full
    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)
