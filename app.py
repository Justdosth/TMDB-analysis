from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load the movie dataset
movies_df = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits_df = pd.read_csv('dataset/tmdb_5000_credits.csv')

# Combine relevant columns into a single string for each movie
credits_df['tags'] = movies_df['genres'] + ' ' + credits_df['cast'] + ' ' + credits_df['crew']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the tags into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(credits_df['tags'])

# Calculate cosine similarity between movie vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the input movie title from the form
        input_movie_title = request.form['movie_title']
        
        # Find the index of the input movie in the dataset
        idx = movies_df[movies_df['title'] == input_movie_title].index[0]

        # Get the similarity scores for the input movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of top 5 similar movies
        sim_indices = [i[0] for i in sim_scores[1:6]]

        # Get the titles of recommended movies
        recommended_movies = movies_df['title'].iloc[sim_indices].tolist()

        return render_template('recommend.html', movie_title=input_movie_title, recommended_movies=recommended_movies)

    except IndexError:
        # Redirect to the apology page if no similar movies are found
        return redirect('/apology')

@app.route('/apology')
def apology():
    return render_template('apology.html')

if __name__ == '__main__':
    app.run(debug=True)
