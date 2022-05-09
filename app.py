import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components

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

#Gets the poster of the movie
def get_poster(title):
  title = title.replace(" ", "+")
  print(title)
  url = 'https://api.themoviedb.org/3/search/movie?api_key=02677c02e0ecdad391d2da9f8943cd61&query=' + title
  print(url)
  response = requests.get(url)
  data = response.json()
  url = "https://image.tmdb.org/t/p/w500/" + data['results'][0]['poster_path']
  return url

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

st.title("Highly Acclaimed Movie Recommendation System")


selected_movie = st.selectbox('Pick a movie that you like',
movies_list['Series_Title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
      st.text(recommendations[0])
      st.image(get_poster(recommendations[0]))
    with col2:
      st.text(recommendations[1])
      st.image(get_poster(recommendations[1]))
    with col3:
      st.text(recommendations[2])
      st.image(get_poster(recommendations[2]))
    with col4:
      st.text(recommendations[3])
      st.image(get_poster(recommendations[3]))
    with col1:
      st.text(recommendations[4])
      st.image(get_poster(recommendations[4]))
    with col2:
      st.text(recommendations[5])
      st.image(get_poster(recommendations[5]))
    with col3:
      st.text(recommendations[6])
      st.image(get_poster(recommendations[6]))      



genre_list = ["Action", "Adventure", "Animation", "Biography",
              "Comedy", "Crime", "Drama", "Family",
              "Fantasy", "Film-Noir", "Horror", "Musical",
              "Mystery", "Romance", "Sci-Fi", "Thriller",
              "War", "Western"]
genre_count = []
for y in genre_list:
  count = 0
  for x in movies["Genre"]:
    if y in x:
      count += 1
  genre_count.append(count)


if st.button("See total movies based on genre"):
  c = (Bar()
      .add_xaxis(genre_list)
      .add_yaxis('Hover on bar to see genre', genre_count)
      .set_global_opts(title_opts=opts.TitleOpts(title="Total Movies By Genre"))
      .render_embed() # generate a local HTML file
  )
  components.html(c, width=1000, height=1000)