import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(ratings.shape)
train, test = train_test_split(ratings, test_size=0.2, random_state=35)

n_users = ratings.userId.nunique()
print('n_users =', n_users)

n_movies = ratings.movieId.nunique()
print('n_movies =', n_movies)

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

model.fit([train.userId, train.movieId], train.rating, epochs=5, verbose=True)

history = model.fit([train.userId, train.movieId], train.rating, epochs=5, verbose=True)
model.save('MovieLensModel')
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Training Error")

predictions = model.predict([test.userId.head(10), test.movieId.head(10)])

[print(predictions[i], test.rating.iloc[i]) for i in range(0, 10)]
