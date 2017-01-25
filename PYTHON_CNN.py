'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
import csv
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#print(X_train.shape)

matFiles = []
trainTestIDs = []
trainTestLabels = []

validationIDs = []

with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

with open('stage1_sample_submission.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validationIDs.append(row['id'])


#TODO: CHANGE WHEN DONE TESTING
#numTrainTest = len(trainTestIDs);
numTrainTest=100
#numValid = len(validationIDs)
numValid=10

#numFeats = 256*256*100
Xdata = np.zeros((numTrainTest,256,256,100))
Ydata = np.zeros(numTrainTest)

Xvalid = np.zeros((numValid,256,256,100))

for ind in range(numTrainTest):
    patID = trainTestIDs[ind]
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_"+patID+".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    #volDataVector = np.reshape(volData,(numFeats))
    Xdata[ind, :,:,:] = volData
    Ydata[ind] = int(trainTestLabels[ind])
    print("Ind:" + str(ind))

for ind in range(numValid):
    patID = validationIDs[ind]
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_"+patID+".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    #volDataVector = np.reshape(volData,(numFeats))
    Xvalid[ind, :,:,:] = volData
    print("Ind2:" + str(ind))

Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xdata,Ydata,test_size=0.1,random_state=42)

# clf = RandomForestClassifier(max_depth=5, n_estimators=20, max_features=100)
# clf = clf.fit(Xtrain,Ytrain)
# yHatTrainP = clf.predict_proba(Xtrain)
# yHatTestP = clf.predict_proba(Xtest)
# YvalidP = clf.predict_proba(Xvalid)

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
input_shape = (256,256,100)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = Xtrain.astype('float32')
X_test = Xtest.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Ytrain, nb_classes)
Y_test = np_utils.to_categorical(Ytest, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""
sio.savemat('RES_randomProj.mat',mdict={'yHatTrainP':yHatTrainP,'yHatTestP':yHatTestP,'YvalidP':YvalidP,'Ytrain':Ytrain,'Ytest':Ytest})

Yvalid = YvalidP[:,1]

with open('submissionRandProjRandomForest.csv', 'w') as csvfile:
    fieldnames = ['id', 'cancer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(numValid):
        writer.writerow({'id': validationIDs[ind], 'cancer': str(Yvalid[ind])})

"""