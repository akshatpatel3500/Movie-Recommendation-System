from flask import Flask, request, render_template
import boto3
import os
from pyspark.sql import SparkSession
import boto3
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.ml.recommendation import ALSModel

spark = SparkSession.builder \
    .appName("MovieLens Recommender System") \
    .getOrCreate()

aws_access_key_id = 'AKIAR4WISA2TVKQGYDF7'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

bucket_name = 'movielens-pyspark'
response = s3.get_object(Bucket=bucket_name, Key="small/movies.csv")
movies_df = pd.read_csv(response['Body'])
movies_df = spark.createDataFrame(movies_df)
response = s3.get_object(Bucket=bucket_name, Key="small/ratings.csv")
ratings_df = pd.read_csv(response['Body'])
ratings_df = spark.createDataFrame(ratings_df)


model_path = "saved_model/"
model = ALSModel.load(model_path)


def recommendMovies(model, user, nbRecommendations):
    dataSet = ratings_df.select('movieId').distinct().withColumn('userId', lit(user))
    moviesAlreadyRated = ratings_df.filter(ratings_df.userId == user).select('movieId', 'userId')
    predictions = model.transform(dataSet.subtract(moviesAlreadyRated)).dropna().orderBy('prediction', ascending=False).limit(nbRecommendations).select('movieId', 'prediction')    
    recommendations = predictions.join(movies_df, predictions.movieId == movies_df.movieId).select(predictions.movieId, movies_df.title, movies_df.genres, predictions.prediction)
    return recommendations

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['userid']
    try:
        user_id = int(user_id)
        recommendations = recommendMovies(model, user_id, 10)
        return recommendations.toPandas().to_html(header="true", table_id="table")  # Convert to HTML table
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
