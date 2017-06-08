import pandas
import os
import numpy as np
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Dense, Flatten, Reshape, Merge, Dropout, Add,Dot,Input,Concatenate
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import json
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn import manifold
import sys

# Read in the dataset
RNG_SEED = 1446557
directory = sys.argv[1]

userscsv = os.path.join(directory, 'users.csv')
traincsv = os.path.join(directory, 'train.csv')
moviecsv = os.path.join(directory, 'movies.csv')
testcsv = os.path.join(directory, 'test.csv')
 
users = pandas.read_csv(userscsv, sep='::', 
                        engine='python', 
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],skiprows=[0]).set_index('UserID')

ratings = pandas.read_csv(traincsv, engine='python', 
                          sep=',', names=['TrainDataID','UserID', 'MovieID', 'Rating'],skiprows=[0])
shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
                          
movies = pandas.read_csv(moviecsv, engine='python',
                         sep='::', names=['movieID', 'Title', 'Genres'],skiprows=[0]).set_index('movieID')
movies['Genres'] = movies.Genres.str.split('|')
movies_genres = np.array(movies['Genres'])


# Count the movies and users
n_movies = movies.shape[0]
n_users = users.shape[0]

#
shuffled_ratings.MovieID = pandas.to_numeric(shuffled_ratings.MovieID, errors='coerce')
movieid = np.array(shuffled_ratings.MovieID)
shuffled_ratings.UserID = pandas.to_numeric(shuffled_ratings.UserID, errors='coerce')
userid = np.array(shuffled_ratings.UserID)



# y_data
shuffled_ratings.Rating = pandas.to_numeric(shuffled_ratings.Rating, errors='coerce')
y = np.array(shuffled_ratings.Rating)
train_ratings_mean = np.mean(y)
train_ratings_std = np.std(y)
# if normalized
#y = (y-train_ratings_mean)/train_ratings_std

def draw(x):
	x = np.array(x,dtype = np.float64)
  
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	vis_data = tsne.fit_transform(x)
	vis_x = vis_data[:, 0]
	vis_y = vis_data[:, 1]

	cm = plt.cm.get_cmap('RdYlBu')
	for i in range(vis_data.shape[0]):
	    if (i<movies_genres.shape[0]): 
	        if (('Thriller' in movies_genres[i])or('Horror' in movies_genres[i])or('Crime' in movies_genres[i])): 
	            plt.scatter(vis_x[i], vis_y[i], c='turquoise', cmap=cm)
	        elif (('Adventure' in movies_genres[i])or('Animation' in movies_genres[i])or("Children's" in movies_genres[i])):
	            plt.scatter(vis_x[i], vis_y[i], c='cornflowerblue', cmap=cm)
	        elif (('Drama' in movies_genres[i])or('Musical' in movies_genres[i])):           
	            plt.scatter(vis_x[i], vis_y[i], c='gold', cmap=cm)
	#sc = plt.scatter(vis_x, vis_y, c=y[1], cmap=cm)
	#plt.colorbar(sc)
	plt.show()


# MF model
def get_model(n_users, n_items, latent_dim):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users+100, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	user_vec = Dropout(0.1)(user_vec)
	item_vec = Embedding(n_items+300, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	item_vec = Dropout(0.1)(item_vec)
	user_bias = Embedding(n_users+100,1, embeddings_initializer='zeros')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items+300, 1, embeddings_initializer='zeros')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes=1)([user_vec,item_vec])
	r_hat = Add()([r_hat,user_bias, item_bias])
	model = keras.models.Model([user_input,item_input],r_hat)
	model.compile(loss='mse',optimizer='adamax')
	return model
 
# DNN model
def nn_model(n_users, n_items, latent_dim):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	user_vec = Embedding(n_users+100, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items+300, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	merge_vec = Concatenate()([user_vec,item_vec])
	hidden = Dropout(0.1)(merge_vec)
	hidden = Dense(120,activation='relu')(hidden)
	hidden = Dropout(0.1)(hidden)
	output = Dense(1,activation='linear')(hidden)	
	model = keras.models.Model([user_input,item_input],output)
	model.compile(loss='mse',optimizer='adamax')
	return model 



model = get_model(n_users, n_movies, 20)
print(model.summary())

'''
earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
                                 
history=model.fit([userid,movieid], y,epochs=1000, validation_split=0.1,batch_size=128,callbacks=[earlystopping,checkpoint],verbose=1)


model.save('./model')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''

# record training history
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''

'''
# get movie embedding 
movie_emb = np.array(loaded_model.layers[3].get_weights()).squeeze()
print('movie embedding shape:',movie_emb.shape)
np.save('movie_emb.npy',movie_emb)

# draw movie embedding
draw(movie_emb)
'''

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best.hdf5")
print("Loaded model from disk")

# read in test data
testing_dataframe = pandas.read_csv(testcsv)
testing_dataset = testing_dataframe.values

test_Movies = testing_dataset[:,2]
test_Users = testing_dataset[:,1]

# predict
ans = loaded_model.predict([test_Users.reshape(-1,1),test_Movies.reshape(-1,1)],batch_size=128)

with open(sys.argv[2],'w') as f:
    f.write('TestDataID,Rating\n')
    for idx, a in enumerate(ans):
        #normalized
        #f.write('{},{}\n'.format(idx+1,(a[0]*train_ratings_std)+train_ratings_mean))
        f.write('{},{}\n'.format(idx+1,a[0]))