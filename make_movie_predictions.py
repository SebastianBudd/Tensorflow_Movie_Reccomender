import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pickle

ratings = pd.read_csv('ratings2.csv')
print(ratings.shape)
train, test = train_test_split(ratings, test_size=0.2, random_state=6)

n_users = ratings.userId.nunique()
print('n_users =', n_users)

n_movies = ratings.movieId.nunique()
print('n_movies =', n_movies)

# map movie ids to a list of integers from 0 to 9723
uniqueIds = ratings.movieId.unique()
print(uniqueIds)

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
print(movie_data[:5])

user = np.array([1 for i in range(len(movie_data))])
print(user[:5])

predictions = model.predict([user, movie_data])

predictions = np.array([a[0] for a in predictions])
print(predictions)
recommended_movie_ids = backwards[(-predictions).argsort()[:5]]

print(recommended_movie_ids)

# print predicted scores
print(predictions[recommended_movie_ids])

movies = pd.read_csv('ml-latest-small/movies.csv')
print(movies.head())

print(movies[movies['id'].isin(recommended_movie_ids)])
