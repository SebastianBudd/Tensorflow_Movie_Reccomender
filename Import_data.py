#  This script will read the data and then perform some exploratory data analysis
import pandas as pd
import pickle

links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

print(links, movies, ratings, tags)

n_users = len(ratings.userId.unique())
print('n_users =', n_users)

n_unique_tags = len(tags.tag.unique())
print('n_unique_tags =', n_unique_tags)

n_movies = ratings.movieId.nunique()
print('n_movies =', n_movies)

# map movie ids to a list of integers from 0 to 9723
uniqueIds = ratings.movieId.unique()
print(uniqueIds)

forwards = {}
backwards = {}
for x in range(n_movies):
    forwards.update({uniqueIds[x]: x})
    backwards.update({x: uniqueIds[x]})

# save these dictionaries as json for use in other scripts
f_file = open('forwards_dict.pkl', 'wb')
pickle.dump(forwards, f_file)
f_file.close()

b_file = open('backwards_dict.pkl', 'wb')
pickle.dump(backwards, b_file)
b_file.close()

ratings2 = ratings
# convert pandas data frame
for j in range(len(ratings)):
    ratings2.iloc[j, 1] = forwards[ratings.iloc[j, 1]]

ratings2.to_csv('ratings2.csv')
