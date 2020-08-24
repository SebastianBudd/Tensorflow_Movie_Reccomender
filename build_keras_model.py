import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
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

print(forwards)
print(type(forwards))

# creating movie embedding path
movie_input = Input(shape=[1], name="Movie-Input")
movie_embedding = Embedding(n_movies + 1, 5, name="Movie-Embedding")(movie_input)
movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([movie_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model = Model([user_input, movie_input], out)
print(model.summary())
print('')
model.compile('adam', 'mean_squared_error')

history = model.fit([train.userId, train.movieId], train.rating, epochs=50, verbose=True)
model.save('MovieLensModel')
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Training Error")

predictions = model.predict([test.userId.head(10), test.movieId.head(10)])

[print(predictions[i], test.rating.iloc[i]) for i in range(0, 10)]

results = model.evaluate([test.userId, test.movieId], test.rating)
print("Evaluate on test data")
print("test loss, test acc:", results)
