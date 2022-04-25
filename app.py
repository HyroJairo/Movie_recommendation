import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#Read the list
movies = pd.read_csv('imdb_top_1000.csv')

#Create a list of important columns for the recommendation engine
columns = ['Actors', 'Director', 'Genre', 'Series_Title']

#Create a function to combine the values of the important columns into a single string
def get_important_features(data):
  important_features = []
  for i in range(0, data.shape[0]):
    important_features.append(data['Actors'][i] + ' ' + data['Director'][i] + ' ' + data['Genre'][i] + ' ' + data['Series_Title'][i])

  return important_features

#Create a column to hold the combined strings
movies['important_features'] = get_important_features(movies)

#Convert the text to a matrix of token counts
cm = CountVectorizer().fit_transform(movies['important_features'])

#Get the cosine similarity matrix from the count matrix
cs = cosine_similarity(cm)

#Get the title of the movie that the user likes
def recommend(title):
    #Find the movie id
    movie_id = movies[movies.Series_Title == title]['Movie_id'].values[0]
    #Create a list of enumerations for the similarity score
    scores = list(enumerate(cs[movie_id]))
    #Sort the list
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
    sorted_scores = sorted_scores[1:]
    #Create a loop to print the first 7 similar movies
    j= 0
    movies_list = []
    for item in sorted_scores:
        movie_title = movies[movies.Movie_id == item[0]]['Series_Title'].values[0]
        movies_list.append(movie_title)
        j = j + 1
        if j > 6:
            break
    return movies_list

movies_list = pickle.load(open('movies.pkl', 'rb'))
movies_list = pd.DataFrame(movies_list)

st.title("Movie Recommendation System")


selected_movie = st.selectbox('Pick a movie that you like',
movies_list['Series_Title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)