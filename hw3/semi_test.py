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

def read_test_dataset(test_file):
    datas = []

    with open(test_file) as file:
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

test_feats= read_test_dataset(sys.argv[1])

emotion_classifier = load_model('./semi_model')
emotion_classifier.summary()
ans = emotion_classifier.predict_classes(test_feats,batch_size=128)

with open(sys.argv[2],'w') as f:
    f.write('id,label\n')
    for idx, a in enumerate(ans):
        f.write('{},{}\n'.format(idx,a))