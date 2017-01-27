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
from keras.layers import Convolution3D, MaxPooling3D
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
numTrainTestAll = len(trainTestIDs);
numTrainTest=100
numValid = len(validationIDs)
#numValid=10

randInds = np.random.permutation(numTrainTestAll)
randIndsUse = randInds[0:numTrainTest]

#numFeats = 256*256*100
Xdata = np.zeros((numTrainTest,256,256,100))
Ydata = np.zeros(numTrainTest)

Xvalid = np.zeros((numValid,256,256,100))

for ind in range(numTrainTest):
    patID = trainTestIDs[randIndsUse[ind]]
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

batch_size = 6
nb_classes = 2
nb_epoch = 5

# input image dimensions
img_rows = 256
img_cols = 256
img_sli = 100

# number of convolutional filters to use
nb_filters = 10
# size of pooling area for max pooling
pool_size = (5,5,5)
# convolution kernel size
kernel_size = (4,4,4)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, img_rows, img_cols,img_sli)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, img_rows, img_cols,img_sli)
    Xvalid = Xvalid.reshape(Xvalid.shape[0], 1, img_rows, img_cols, img_sli)
    input_shape = (1, img_rows, img_cols,img_sli)
else:
    Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols,img_sli, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols,img_sli, 1)
    Xvalid = Xvalid.reshape(Xvalid.shape[0], img_rows, img_cols, img_sli, 1)
    input_shape = (img_rows, img_cols,img_sli, 1)

#Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols,img_sli,1)
#Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols,img_sli,1)
#input_shape = (img_rows, img_cols,img_sli,1)

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
#Xtrain /= 255
#Xtest /= 255
print('X_train shape:', Xtrain.shape)
print(Xtrain.shape[0], 'train samples')
print(Xtest.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Ytrain, nb_classes)
Y_test = np_utils.to_categorical(Ytest, nb_classes)

print(Y_test)


#Here is code from a 3D CNN example on the following blog:
#   http://learnandshare645.blogspot.in/2016/06/3d-cnn-in-keras-action-recognition.html
#
#Good initial CNN tutorial:
#   http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

model = Sequential()
model.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1],kernel_size[2],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(MaxPooling3D(pool_size=pool_size))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, init='normal',activation='relu'))
model.add(Dense(nb_classes, init='normal',activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

#ERROR HERE. TODO: LOOK AT THESE TWO PAGES
#   https://github.com/fchollet/keras/issues/3009
#   https://github.com/fchollet/keras/issues/3109

model.fit(Xtrain, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(Xtest, Y_test))
score = model.evaluate(Xtest, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

newPred = model.predict(Xvalid)
sio.savemat('CNN_currentPred.mat',mdict={'newPred':newPred})

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