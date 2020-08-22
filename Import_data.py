#  This script will read the data and then perform some exploratory data analysis
import pandas as pd

links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

print(links, movies, ratings, tags)

n_users = len(ratings.userId.unique())
print('n_users =', n_users)

n_unique_tags = len(tags.tag.unique())
print('n_unique_tags =', n_unique_tags)
