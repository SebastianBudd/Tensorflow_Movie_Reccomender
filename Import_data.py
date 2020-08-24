#  This script will read the data and then perform some exploratory data analysis
import pandas as pd
import pickle

ratings = pd.read_csv('ml-latest-small/ratings.csv')

print(ratings)

n_users = len(ratings.userId.unique())
print('n_users =', n_users)

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
print("Forwards dictionary saved")

b_file = open('backwards_dict.pkl', 'wb')
pickle.dump(backwards, b_file)
b_file.close()
print("Backwards dictionary saved")

ratings2 = ratings
# convert pandas data frame
for j in range(len(ratings)):
    ratings2.iloc[j, 1] = forwards[ratings.iloc[j, 1]]

ratings2.to_csv('ratings2.csv')
