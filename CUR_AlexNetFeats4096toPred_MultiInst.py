from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
from scipy.misc import imread, imresize, imsave
import time
import datetime

from sklearn.decomposition import PCA,KernelPCA
from convnetskeras.convnets import preprocess_image_batch, convnet
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn import datasets, linear_model
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids
import csv

batch_size = 20
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows = 256
img_cols = 256
img_sli = 100

# number of convolutional filters to use
nb_filters = 10
# size of pooling area for max pooling
pool_size = (5,5,5)
pool_size2 = (5,5)
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

numTrainTest = len(trainTestIDs)
numValid = len(validationIDs)

XvalidOld = np.zeros((numValid, 100, 4096))
xTrainTestOld = np.zeros((numTrainTest, 100, 4096))
yTrainTest = np.zeros((numTrainTest))

Xvalid2 = np.zeros((numValid*100,4096))
xTrainTest2 = np.zeros((numTrainTest*100,4096))

print('Loading Train/Test Set')
for kk in range(len(trainTestIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+trainTestIDs[kk]+'.npy'
    curX = np.load(curFile)
    xTrainTestOld[kk, :, :] = curX
    yTrainTest[kk] = trainTestLabels[kk]
    startInd = kk * 100
    endInd = (kk + 1) * 100
    xTrainTest2[startInd:endInd, :] = curX

print('Loading Validation Set')
for kk in range(len(validationIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+validationIDs[kk]+'.npy'
    curX = np.load(curFile)
    XvalidOld[kk, :, :] = curX
    startInd = kk * 100
    endInd = (kk + 1) * 100
    Xvalid2[startInd:endInd, :] = curX


encoding_dim = 64
input_img = Input(shape=(4096,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(4096, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(xTrainTest2, xTrainTest2,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(Xvalid2, Xvalid2))
encoder = Model(input=input_img, output=encoded)
validationEncoded = encoder.predict(Xvalid2)
trainTestEncoded = encoder.predict(xTrainTest2)

XvalidNew = np.zeros((numValid,100,encoding_dim))
xTrainTestNew = np.zeros((numTrainTest,100,encoding_dim))
print('Loading Train/Test Set Encoded')
for kk in range(len(trainTestIDs)):
    startInd = kk * 100
    endInd = (kk + 1) * 100
    xTrainTestNew[kk,:,:] = trainTestEncoded[startInd:endInd, :]

print('Loading Validation Set Encoded')
for kk in range(len(validationIDs)):
    startInd = kk * 100
    endInd = (kk + 1) * 100
    XvalidNew[kk, :, :] = validationEncoded[startInd:endInd, :]

#TODO: CHANGE THIS BLOCK IF WANT TO SWITCH BACK TO 4096-DIM
numDim=4096
Xvalid=XvalidOld
xTrainTest=xTrainTestOld

inputI = Input(shape=(1,100,numDim))
layer1 = Convolution2D(2,1,numDim)(inputI)
layer2 = Activation('sigmoid')(layer1)
layer3 = MaxPooling2D(pool_size=(100,1),border_mode='valid')(layer2)
layer4 = Flatten()(layer3)
model2 = Model(input=inputI, output=layer4)


print('Separating Training and Test Data')
Xtrain,Xtest,Ytrain,Ytest = train_test_split(xTrainTest,yTrainTest,test_size=0.1,random_state=42)

Xtrain = Xtrain.reshape(Xtrain.shape[0], 1,100,numDim)
Xtest = Xtest.reshape(Xtest.shape[0], 1,100,numDim)
Xvalid = Xvalid.reshape(Xvalid.shape[0], 1,100,numDim)
xTrainTest = xTrainTest.reshape(xTrainTest.shape[0], 1,100,numDim)

Ytest2 = np_utils.to_categorical(Ytest, nb_classes)
Ytrain2 = np_utils.to_categorical(Ytrain, nb_classes)

model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.fit(Xtrain, Ytrain2,
                nb_epoch=20,
                batch_size=256,
                shuffle=True,
                validation_data=(Xtest, Ytest2))

patientPrediction = model2.predict(Xvalid)
patientTrainTest = model2.predict(xTrainTest)
ptPrediction = patientPrediction[:, 1]
ptTrainTest = patientTrainTest[:,1]
"""
np.save('temp/slicePrediction.npy',slicePrediction)
np.save('temp/sliceTrainTest.npy',sliceTrainTest)

slicePrediction = np.load('temp/slicePrediction.npy')
sliceTrainTest = np.load('temp/sliceTrainTest.npy')

mseTrainTest = ((ptTrainTest-trainTestLabels)**2).mean()

print('Training,Test MSE:' + str(mseTrainTest))
"""


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'submissions/kagglePredFrom_'+st+'.csv'

with open(fileName, 'w') as csvfile:
    fieldnames = ['id', 'cancer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(numValid):
        writer.writerow({'id': validationIDs[ind], 'cancer': str(ptPrediction[ind])})
