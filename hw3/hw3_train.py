import csv
import os
import numpy as np 
import sys
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation,BatchNormalization
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
nb_classes = 7
# read train_data
def read_dataset(train_file,isFeat=True):
    datas = []
    with open(train_file) as file:
        next(file)
        for line_id,line in enumerate(file):
            label, feat=line.split(',')
            feat = np.fromstring(feat,dtype='float32',sep=' ')
            feat = (feat-128)/255
            #print(feat)
            feat = np.reshape(feat,(48,48,1))

            datas.append((feat,int(label),line_id))

    #random.shuffle(datas)  # shuffle outside
    feats,labels,line_ids = zip(*datas)
    feats = np.asarray(feats)
    labels = to_categorical(np.asarray(labels,dtype=np.int32))
    return feats,labels,line_ids



def read_test_dataset(mode='test'):
    datas = []

    with open(os.path.join('./','{}.csv'.format(mode))) as file:
        next(file)
        for line_id,line in enumerate(file):
            _,feat = line.split(',')
            feat = np.fromstring(feat,dtype='float32',sep=' ')
            feat = (feat-128)/255
            #print(feat)
            feat = np.reshape(feat,(48,48,1))
            datas.append(feat)

    #random.shuffle(datas)  # shuffle outside

    feats = datas
    feats = np.asarray(feats)
    return feats
    
def add_unlabel_featrue2training(training_feature, training_yhat, unlabel_feature, model, confidence_th):
	unlabel_prediction = model.predict(unlabel_feature)
	# print("nb_unlabel_feature:" + str(len(unlabel_feature) ))
	# print("nb_unlabel_prediction:" + str(len(unlabel_prediction)))
	nb_unlabel_data = len(unlabel_prediction)

	delete_array = np.array([0]*nb_unlabel_data).astype(dtype='int')
	delete_index = np.array([]).astype(dtype='int')
	count = 0

	print("[ original data ] : training->" + str(len(training_feature)) + "  unlabel->"+str(len(unlabel_feature)) )


	for i in range(nb_unlabel_data):
		if(  np.amax( unlabel_prediction[i] ) >= confidence_th  ):

			delete_array[i] = 1
			max_index = np.argmax(unlabel_prediction[i])
			tmp_prediction = unlabel_prediction[i]
			tmp_prediction.fill(0)
			tmp_prediction[max_index] = 1.0

			tmp_feature =  unlabel_feature[i]
			tmp_feature = np.array( tmp_feature ).astype(dtype='float32').reshape(1,48,48,1)
			#print("!!!!!!!!!!!!!!!!!!",training_feature.shape )
			#print("!!!!!!!!!!!!!!!!!!",tmp_feature.shape )
			training_feature = np.vstack((training_feature, tmp_feature ))


			tmp_prediction = np.array( tmp_prediction ).astype(dtype='float32').reshape(1,7)			
			training_yhat= np.vstack((training_yhat, tmp_prediction ))

			count += 1
		if(i%1000 == 0 ):
			print("check unlabel confidence :" + str(i) +"  data")


	for i in range( nb_unlabel_data):
		if(delete_array[i] == 1):
			delete_index = np.append(delete_index, i )	
		if(i%1000 == 0 ):
			print("delete unlabel feature :" + str(i) +"  data")

	unlabel_feature = np.delete(unlabel_feature, delete_index, axis = 0 )	


	print("after confidence similar trimming:")
	print("total move :" + str(count) + "  data from unlabel to training ")
	print("[after data] : training->" + str(len(training_feature)) + "  unlabel->" + str(len(unlabel_feature)) )



	return training_feature, training_yhat, unlabel_feature  

def split_data(training_feature, training_yhat, validation_percent):
	
	training_percent = 1 - validation_percent 
	num_training = int(training_percent * training_feature.shape[0])
	indices = np.random.permutation(training_feature.shape[0])
	training_idx,validation_idx = indices[:num_training], indices[num_training:]
	#print("training_feature.shape[0]",training_feature.shape[0])
	#print("training_yhat.shape[0]",training_yhat.shape[0])
	#print()

	training_feature ,validation_feature = training_feature[training_idx,:], training_feature[validation_idx,:]
	training_yhat ,validation_yhat = training_yhat[training_idx,:], training_yhat[validation_idx,:]

	return training_feature , training_yhat , validation_feature , validation_yhat       
         
def build_model(mode):
    model = Sequential()
    if mode == 'easy':
    # CNN part (you can repeat this part several times)
        model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(48,48,1)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        opt = SGD(lr=0.01,decay=1e-6,momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary() # show the whole model in terminal
    return model

datagen = ImageDataGenerator(
            featurewise_center = False,  # set input mean to 0 over the dataset
            samplewise_center = False,  # set each sample mean to 0
            featurewise_std_normalization = False,  # divide inputs by std of the dataset
            samplewise_std_normalization = False,  # devide each input by its std
            zca_whitening = False,  # apply ZCA whitening
            rotation_range = 12,  # randomly rotate images in range
            width_shift_range = 0.2,  # randomly shift images horizontally
            height_shift_range = 0.2,  # randomly shift images vertically
            horizontal_flip = True,  # randomly flip images
            vertical_flip = False)  # randomly flip images


tr_feats, tr_labels, _ = read_dataset(sys.argv[1])
#dev_feats, dev_labels,_ = read_dataset('valid')
#test_feats= read_test_dataset('test')

datagen.fit(tr_feats)
#(training_feature , training_yhat , validation_feature , validation_yhat) = split_data(tr_feats , tr_labels , 0.1 )



#onfidence_th = 0.995
#unlabel data
#unlabel_feature = test_feats


#early stop
#early_stop = EarlyStopping(monitor='val_loss' , patience=20, verbose=1)

#nb_add_unlabel = 2

#for i in range(nb_add_unlabel):	
#    if i == nb_add_unlabel -1:
#        emotion_classifier = build_model('easy')
#        emotion_classifier.fit(training_feature, training_yhat, batch_size=128, nb_epoch=1,validation_data=(validation_feature, validation_yhat), callbacks=[early_stop])
#    else:
#        emotion_classifier = build_model('easy')
#        emotion_classifier.fit(training_feature, training_yhat, batch_size=128, nb_epoch =1, validation_data=(validation_feature, validation_yhat), callbacks=[early_stop] )
#        (training_feature, training_yhat, unlabel_feature) = add_unlabel_featrue2training(training_feature, training_yhat, unlabel_feature, emotion_classifier, confidence_th)
emotion_classifier = build_model('easy')
emotion_classifier.fit_generator(datagen.flow(tr_feats, tr_labels,
                            batch_size = 128),
                            samples_per_epoch = tr_feats.shape[0],
                            nb_epoch = 100)    
    
emotion_classifier.save('model')
