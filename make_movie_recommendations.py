import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pickle
import requests

ratings = pd.read_csv('ratings2.csv')
# print(ratings.shape)
train, test = train_test_split(ratings, test_size=0.2, random_state=6)

n_users = ratings.userId.nunique()
# print('n_users =', n_users)

n_movies = ratings.movieId.nunique()
# print('n_movies =', n_movies)

# map movie ids to a list of integers from 0 to 9723
uniqueIds = ratings.movieId.unique()
# print(uniqueIds)

# load dictionaries to convert movie Ids to a list of consecutive integers and back to the original ids
with open('forwards_dict.pkl', 'rb') as f_file:
    forwards = pickle.load(f_file)
    f_file.close()

with open('backwards_dict.pkl', 'rb') as b_file:
    backwards = pickle.load(b_file)
    b_file.close()

model = keras.models.load_model('MovieLensModel')

# Creating dataset for making recommendations for the first user
movie_data = np.array(list(set(ratings.movieId)))
# print(movie_data[:5])

user = np.array([16 for i in range(len(movie_data))])
# print(user[:5])

predictions = model.predict([user, movie_data])

predictions = np.array([a[0] for a in predictions])
recommended_movie_ids = (-predictions).argsort()[:5]

recommended_movie_ids2 = []
for x in range(len(recommended_movie_ids)):
    recommended_movie_ids2.append(backwards[recommended_movie_ids[x]])

# print predicted scores


# print movie recommendations
movies = pd.read_csv('ml-latest-small/movies.csv')
links = pd.read_csv('ml-latest-small/links.csv')
rmovies = movies[movies['movieId'].isin(recommended_movie_ids2)]
rlinks = links[movies['movieId'].isin(recommended_movie_ids2)]


recommended_movie_df = pd.DataFrame(columns=['movieId', 'Title', 'tmdbId', 'Predicted Rating'])
for movie in range(5):
    api_key = "32ad08749394fca3d4f58d4fcd0edfe7"
    url = 'https://api.themoviedb.org/3/movie/{}?api_key={}'.format(rlinks.iloc[movie, 2], api_key)
    results = requests.get(url).json()
    Title = results['title']
    predicted_rating = (predictions[recommended_movie_ids])[movie]
    new_row = {'movieId': rmovies.iloc[movie, 0], 'Title': Title, 'tmdbId': rlinks.iloc[movie, 2], 'Predicted Rating': predicted_rating}
    recommended_movie_df = recommended_movie_df.append(new_row, ignore_index=True)

recommended_movie_df.to_csv('user16recommendations.csv')
