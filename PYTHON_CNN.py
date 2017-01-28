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
import time
import datetime
import csv
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


batch_size = 10
nb_classes = 2
nb_epoch = 2

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
trainingRatio = 0.90
numTrainTestAll = len(trainTestIDs)
#numTrainTestAll = 100

numTrain = int(np.floor(trainingRatio*numTrainTestAll))
numTest = numTrainTestAll-numTrain
numValid = len(validationIDs)

randInds = np.random.permutation(numTrainTestAll)
indsTrain = randInds[0:numTrain]
indsTest = randInds[numTrain:numTrainTestAll]

Xtrain = np.zeros((numTrain,256,256,100))
Ytrain = np.zeros(numTrain)
Xtest = np.zeros((numTest,256,256,100))
Ytest = np.zeros(numTest)


Xvalid = np.zeros((numValid,256,256,100))

def getVolData(patID):
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_" + patID + ".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    return volData

def trainGenerator(trainTestIDs,trainTestLabels,indsTrain):
    while 1:
        for ind in range(len(indsTrain)):
            patID = trainTestIDs[indsTrain[ind]]
            XtrainCur = getVolData(patID)
            if K.image_dim_ordering() == 'th':
                XtrainCur = XtrainCur.reshape(1, 1, img_rows, img_cols, img_sli)
            else:
                XtrainCur = XtrainCur.reshape(1, img_rows, img_cols, img_sli, 1)
            YtrainCur = int(trainTestLabels[indsTrain[ind]])
            YtrainUse = np_utils.to_categorical(YtrainCur, nb_classes)
            print("TrainInd:" + str(ind))
            yield (XtrainCur.astype('float32'),YtrainUse)



for ind in range(numTest):
    patID = trainTestIDs[indsTest[ind]]
    Xtest[ind, :,:,:] = getVolData(patID)
    Ytest[ind] = int(trainTestLabels[indsTest[ind]])
    print("TestInd:" + str(ind))

for ind in range(numValid):
    patID = validationIDs[ind]
    Xvalid[ind, :,:,:] = getVolData(patID)
    print("ValidInd:" + str(ind))

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
#model.add(Dense(128, init='normal',activation='relu'))
model.add(Dense(16, init='normal',activation='relu'))
model.add(Dense(nb_classes, init='normal',activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])

#ERROR HERE. TODO: LOOK AT THESE TWO PAGES
#   https://github.com/fchollet/keras/issues/3009
#   https://github.com/fchollet/keras/issues/3109

model.fit_generator(trainGenerator(trainTestIDs,trainTestLabels,indsTrain),
                    samples_per_epoch = 100, nb_epoch=nb_epoch,
          verbose=1, validation_data=(Xtest, Y_test))
score = model.evaluate(Xtest, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

yValidProb = model.predict_proba(Xvalid,batch_size=batch_size,verbose=1)
yValidPred = model.predict(Xvalid,batch_size=batch_size,verbose=1)
yValidClasses = model.predict_classes(Xvalid,batch_size=batch_size,verbose=1)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'cnnPredictions/cnnPredictionFrom_'+st+'.mat'

sio.savemat(fileName,mdict={'yValidProb':yValidProb,'yValidPred':yValidPred,'yValidClasses':yValidClasses})


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