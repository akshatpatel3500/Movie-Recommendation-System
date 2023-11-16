#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
import boto3
import pandas as pd

aws_access_key_id = 'Redacted'
aws_secret_access_key = 'Redacted'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'movielens-pyspark'


# In[2]:


from pyspark.sql import SparkSession

spark = SparkSession.builder     .appName("MovieLens Recommender System")     .getOrCreate()


# In[3]:


# Replace the file paths with the path to your MovieLens dataset
response = s3.get_object(Bucket=bucket_name, Key="small/movies.csv")
movies_df = pd.read_csv(response['Body'])
movies_df = spark.createDataFrame(movies_df)
response = s3.get_object(Bucket=bucket_name, Key="small/ratings.csv")
ratings_df = pd.read_csv(response['Body'])
ratings_df = spark.createDataFrame(ratings_df)


# In[4]:


# Inspect the first few rows of the dataframes
movies_df.show(5)
ratings_df.show(5)

# Check for missing values
print("Missing values in movies dataset:", movies_df.count() - movies_df.dropna().count())
print("Missing values in ratings dataset:", ratings_df.count() - ratings_df.dropna().count())


# In[5]:


merged_df = movies_df.join(ratings_df, ["movieId"], "inner")

# Show the first few rows of the merged DataFrame to verify the merge
merged_df.show(5)


# In[6]:


from pyspark.sql.functions import countDistinct, col, mean, stddev, min, max

# To count the number of distinct movies and users and display the count
distinct_movies_count = merged_df.agg(countDistinct(col("movieId")).alias("distinct_movies")).show()
distinct_users_count = merged_df.agg(countDistinct(col("userId")).alias("distinct_users")).show()

# To describe specific columns with stats like count, mean, stddev, min, max
merged_df.describe(['rating']).show()

# For numerical columns, we can also calculate the mean, standard deviation, min, and max directly
merged_df.select(
    mean(col("rating")).alias("mean_rating"),
    stddev(col("rating")).alias("stddev_rating"),
    min(col("rating")).alias("min_rating"),
    max(col("rating")).alias("max_rating")
).show()

# To see the distribution of ratings
merged_df.groupBy("rating").count().orderBy("rating").show()


# In[7]:


from pyspark.ml.feature import StringIndexer

# Indexing movie IDs
movie_indexer = StringIndexer(inputCol="movieId", outputCol="movieIdIndexed")
model = movie_indexer.fit(merged_df)
merged_df = model.transform(merged_df)

# Indexing user IDs
user_indexer = StringIndexer(inputCol="userId", outputCol="userIdIndexed")
model = user_indexer.fit(merged_df)
merged_df = model.transform(merged_df)


# In[8]:


from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder     .appName("MovieLens Recommender System")     .getOrCreate()

# Load your merged data (assuming it is already loaded as 'merged_df')

# Casting the userId and movieId to integer type as required by ALS
merged_df = merged_df.withColumn("userId", col("userId").cast("integer"))
merged_df = merged_df.withColumn("movieId", col("movieId").cast("integer"))

# Casting the rating to float as required by ALS
merged_df = merged_df.withColumn("rating", col("rating").cast("float"))

# Handling any missing values if necessary
merged_df = merged_df.na.drop()

# Splitting the data into training and test sets
(training_data, test_data) = merged_df.randomSplit([0.8, 0.2], seed=42)

# Now the data is ready for the ALS model training


# In[9]:


# Initialize the ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)

# Fit the model to the training data
model = als.fit(training_data)

# Apply the model to the test data to make predictions
predictions = model.transform(test_data)


# In[10]:


from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model by computing the RMSE on the test data
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# In[12]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Define the parameter grid
paramGrid = ParamGridBuilder()     .addGrid(als.rank, [10, 50, 100])     .addGrid(als.maxIter, [5, 10, 20])     .addGrid(als.regParam, [0.01, 0.1, 1.0])     .addGrid(als.alpha, [0.01, 0.1, 1.0])     .build()

# Define the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Define the cross-validator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # Use 3+ folds in practice

# Fit the model
cvModel = crossval.fit(training_data)

# Apply the best model to the test data
predictions = cvModel.transform(test_data)

# Evaluate the best model
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# In[13]:


# Display a few prediction results
predictions.select("userId", "movieId", "rating", "prediction").show(5)


# In[14]:


from pyspark.sql.functions import lit

def recommendMovies(model, user, nbRecommendations):
    # Create a DataFrame with the specified user and all the movies listed in the ratings DataFrame
    dataSet = ratings_df.select('movieId').distinct().withColumn('userId', lit(user))
    
    # Create a DataFrame with the movies that have already been rated by this user
    moviesAlreadyRated = ratings_df.filter(ratings_df.userId == user).select('movieId', 'userId')
    
    # Apply the recommender system to the dataset without the already rated movies to predict ratings
    predictions = model.transform(dataSet.subtract(moviesAlreadyRated)).dropna().orderBy('prediction', ascending=False).limit(nbRecommendations).select('movieId', 'prediction')
    
    # Join with the movies DataFrame to get the movies titles and genres
    recommendations = predictions.join(movies_df, predictions.movieId == movies_df.movieId).select(predictions.movieId, movies_df.title, movies_df.genres, predictions.prediction)
    
    return recommendations


# In[15]:


recommendations = recommendMovies(model, 144, 10)
recommendations.show(truncate=False)


# In[16]:


recommendations_for_user_198 = recommendMovies(model, 198, 10)
recommendations_for_user_198.show(truncate=False)


# In[ ]:




