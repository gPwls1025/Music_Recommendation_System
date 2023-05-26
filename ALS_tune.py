import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import Row
from pyspark.sql.functions import avg
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.tuning import ParamGridBuilder

def main(spark, userID, file_size):
    
    #------------------------Load the track, train, val from Parquet files------------------------
    
    print('\nLoading data...\n')
    
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

    #------------------------Change test set to a suitable format------------------------
    
    print('\nConvert test set...\n')
    
    # Join track and interaction table on recording_msid
    joined_df = track_test.join(interactions_test, on="recording_msid", how="inner")

    # If join recording_msid has mbid, replace all of that user's specific msid with mbid
    joined_df = joined_df.withColumn("recording_msid", func.when(joined_df["recording_mbid"].isNotNull(), joined_df["recording_mbid"]).otherwise(joined_df["recording_msid"]))

    # For each user_id, get the count of unique msid.
    test_filtered = joined_df.groupBy("user_id", "recording_msid") \
                             .agg(func.count("recording_msid") \
                             .alias("count")) \
                             .orderBy("user_id", "count", ascending=True)
    
    #------------------------Since ALS takes integer index, convert recording_msid to index------------------------
    
    print('\nIndexing...\n')
    
    # Initialize the StringIndexer
    msid_indexer = StringIndexer(inputCol="recording_msid", outputCol="recording_index")

    # Fit the indexer to training data
    msid_model = msid_indexer.fit(train)

    # Transform training data
    indexed_train = msid_model.transform(train)

    # Transform val data using the same 'msid_model' fitted on the training data
    indexed_val = msid_model.transform(val)

    # Transform test data using the same 'msid_model' fitted on the training data
    indexed_test = msid_model.transform(test_filtered)

    #------------------------ALS Hyperparameter Tuning------------------------
    
    print('\nBegin ALS Hyperparameter Tuning...\n')

    # Initialize the ALS model
    als = ALS(
        rank=150,
        userCol="user_id",
        itemCol="recording_index",
        ratingCol="count",
        coldStartStrategy="drop",
        implicitPrefs=True,
        nonnegative=True
    )

    # pipeline = Pipeline(stages=[als])

    param_grid = ParamGridBuilder() \
                .addGrid(als.alpha, [1, 5, 10]) \
                .build()
                # .addGrid(als.maxIter, [10, 15]) \
                # .addGrid(als.rank, [50, 100, 150]) \
                # .addGrid(als.regParam, [0.0001, 0.001, 0.01, .1]) \
                

    class RankingMetricsEvaluator(Evaluator):
        def __init__(self, k=100, userCol="user_id", itemCol="recording_index", predictionCol="recommendations", als_model=als):
            super(RankingMetricsEvaluator, self).__init__()
            self.k = k
            self.userCol = userCol
            self.itemCol = itemCol
            self.predictionCol = predictionCol
            self.als_model = als_model

        def _evaluate(self, dataset):
            # Update the ALS model's predictionCol attribute
            self.als_model.setPredictionCol("recommendations")

            # Fit the model with the current set of parameters
            als_model = self.als_model.fit(dataset)

            # Generate top-K recommendations for the user subset
            user_subset = dataset.select(self.userCol).distinct()
            user_rec = als_model.recommendForUserSubset(user_subset, self.k)
            
            # Prepare the dataset in a suitable format for RankingMetrics
            indexed_val = dataset.select(col(self.userCol), col(self.itemCol).alias("label"))
            indexed_val = indexed_val.groupBy(self.userCol).agg(expr("collect_list(label) as labels"))

            # Prepare the recommendations in a suitable format for RankingMetrics
            user_rec = user_rec.select(col(self.userCol), col(f"{self.predictionCol}.{self.itemCol}").alias("pred"))
            user_rec = user_rec.join(indexed_val, self.userCol, "inner")

            # Convert predictions and labels columns to RDD format
            user_rec_rdd = user_rec.select("pred", "labels").rdd

            # Instantiate a RankingMetrics object with the prepared data
            metrics = RankingMetrics(user_rec_rdd)
            
            return metrics.meanAveragePrecisionAt(self.k)
        
        def isLargerBetter(self):
            return True
    
    cross_validator = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=RankingMetricsEvaluator(als_model=als),
        numFolds=3
    )

    cv_model = cross_validator.fit(indexed_train)

    print('\Finished ALS Hyperparameter Tuning...\n')

    best_als_model = cv_model.bestModel.stages[0]

    best_maxIter = best_als_model.getMaxIter()
    best_regParam = best_als_model.getRegParam()
    best_rank = best_als_model.getRank()

    print("Best hyperparameters:")
    print("MaxIter:", best_maxIter)
    print("RegParam:", best_regParam)
    print("Rank:", best_rank)

    # Get the average metrics for each combination of hyperparameters
    avg_metrics = cv_model.avgMetrics

    # Create a list of dictionaries containing the hyperparameter combinations
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in param_grid.values()]

    # Create a DataFrame with hyperparameter combinations and their corresponding MAP values
    hyperparameter_results = spark.createDataFrame(
        [Row(**params, MAP=metric) for params, metric in zip(param_combinations, avg_metrics)]
    )

    # Group by 'rank' and calculate the average MAP
    rank_effect = hyperparameter_results.groupBy('rank').agg(avg('MAP').alias('avg_MAP')).orderBy('rank')
    rank_effect.show()

    # Group by 'numIter' and calculate the average MAP
    numIter_effect = hyperparameter_results.groupBy('numIter').agg(avg('MAP').alias('avg_MAP')).orderBy('numIter')
    numIter_effect.show()


if __name__ == "__main__":
    
    # Create the spark session object
    spark = SparkSession.builder.appName('ALS') \
                                .config("spark.sql.broadcastTimeout", "36000") \
                                .config("spark.driver.maxResultSize", "24g") \
                                .config("spark.driver.memory", "32g") \
                                .config("spark.executor.memory", "32g") \
                                .getOrCreate()

    # Get user netID from the command line
    userID = os.environ['USER'] 

    file_size = sys.argv[1]

    # Call our main routine
    main(spark,userID,file_size)
