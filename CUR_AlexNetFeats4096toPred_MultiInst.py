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
Xvalid = np.zeros((numValid*100,4096))
xTrainTest = np.zeros((numTrainTest*100,4096))
yTrainTest = np.zeros((numTrainTest*100))

print('Loading Train/Test Set')
for kk in range(len(trainTestIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+trainTestIDs[kk]+'.npy'
    curX = np.load(curFile)
    startInd = kk*100
    endInd = (kk+1)*100
    xTrainTest[startInd:endInd,:] = curX
    yTrainTest[startInd:endInd] = trainTestLabels[kk]

print('Loading Validation Set')
for kk in range(len(validationIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+validationIDs[kk]+'.npy'
    curX = np.load(curFile)
    startInd = kk*100
    endInd = (kk+1)*100
    Xvalid[startInd:endInd,:] = curX

print('Separating Training and Test Data')
Xtrain,Xtest,Ytrain,Ytest = train_test_split(xTrainTest,yTrainTest,test_size=0.1,random_state=42)

Ytest2 = np_utils.to_categorical(Ytest, nb_classes)
Ytrain2 = np_utils.to_categorical(Ytrain, nb_classes)

input_img = Input(shape=(4096,))
#layer2 = Dense(128,activation='sigmoid')(input_img)
#logisticLayer = Dense(2,activation='sigmoid')(layer2)
logisticLayer = Dense(2,activation='sigmoid')(input_img)

logRegress = Model(input=input_img,output=logisticLayer)
logRegress.compile(optimizer='adadelta', loss='mse')
logRegress.fit(Xtrain, Ytrain2,
                nb_epoch=200,
                batch_size=256,
                shuffle=True,
                validation_data=(Xtest, Ytest2))

slicePrediction = logRegress.predict(Xvalid)
sliceTrainTest = logRegress.predict(xTrainTest)

np.save('temp/slicePrediction.npy',slicePrediction)
np.save('temp/sliceTrainTest.npy',sliceTrainTest)
"""

slicePrediction = np.load('temp/slicePrediction.npy')
sliceTrainTest = np.load('temp/sliceTrainTest.npy')
"""
ptPrediction = np.zeros((numValid))
for kk in range(numValid):
    startInd = kk * 100
    endInd = (kk + 1) * 100
    sliceProbs = slicePrediction[startInd:endInd, 1]
    ptPrediction[kk] = np.max(sliceProbs)

ptTrainTest = np.zeros((numTrainTest))
ptTarget = np.zeros((numTrainTest))
for kk in range(numTrainTest):
    startInd = kk * 100
    endInd = (kk + 1) * 100
    sliceProbs = sliceTrainTest[startInd:endInd, 1]
    ptTrainTest[kk] = np.max(sliceProbs)
    ptTarget[kk] = trainTestLabels[kk]



mseTrainTest = ((ptTrainTest-ptTarget)**2).mean()

print('Training,Test MSE:' + str(mseTrainTest))

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'submissions/kagglePredFrom_'+st+'.csv'

with open(fileName, 'w') as csvfile:
    fieldnames = ['id', 'cancer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(numValid):
        writer.writerow({'id': validationIDs[ind], 'cancer': str(ptPrediction[ind])})

"""
# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(4096,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
#encoded2 = Dense(encoding_dim, activation='sigmoid')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(4096, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(Xtrain, Xtrain,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(Xtest, Xtest))

encoder = Model(input=input_img, output=encoded)
validationEncoded = encoder.predict(Xvalid)
trainTestEncoded = encoder.predict(xTrainTest)

sio.savemat('autoencodingPrediction.mat',
            mdict={'validData':validationEncoded, 'trainTestData':trainTestEncoded})

np.save('temp/validationAutoencoded2.npy',validationEncoded)
np.save('temp/trainTestAutoencoded2.npy',trainTestEncoded)

validationEncoded = np.load('temp/validationAutoencoded2.npy')
trainTestEncoded = np.load('temp/trainTestAutoencoded2.npy')


newTrainTest = np.zeros((numTrainTest,6400))
newValidation = np.zeros((numValid,6400))

print('Making New Train/Test Set')
for kk in range(len(trainTestIDs)):
    startInd = kk*100
    endInd = (kk+1)*100
    newTrainTest[kk,:] = np.reshape(trainTestEncoded[startInd:endInd,:] ,6400)

print('Making New Validation Set')
for kk in range(len(validationIDs)):
    startInd = kk*100
    endInd = (kk+1)*100
    newValidation[kk,:] = np.reshape(validationEncoded[startInd:endInd,:],6400)

XtrainNew,XtestNew,Ytrain2,Ytest2 = train_test_split(newTrainTest,trainTestLabels,test_size=0.1,random_state=42)

Ytest2 = np_utils.to_categorical(Ytest2, nb_classes)
Ytrain2 = np_utils.to_categorical(Ytrain2, nb_classes)


input_img2 = Input(shape=(6400,))
layer1 = Dense(256, init='normal', activation='sigmoid')(input_img2)
layer2 = Dense(128, init='normal', activation='sigmoid')(layer1)
layer3 = Dense(32, init='normal',activation='relu')(layer2)
outputLayer = Dense(nb_classes, init='normal',activation='softmax')(layer3)

post4096Model = Model(input = input_img2,output=outputLayer)
post4096Model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

post4096Model.fit(XtrainNew, Ytrain2, batch_size=500, nb_epoch=50,
                  verbose=1, validation_data=(XtestNew, Ytest2))

prediction = post4096Model.predict(newValidation)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'cnnPredictions/cnnPredictionAlexNet4096From_'+st+'.mat'
sio.savemat(fileName,mdict={'prediction':prediction})



yHatTrainP = regr.predict_proba(Xtrain)
yHatTestP = regr.predict_proba(Xtest)
YvalidP = regr.predict_proba(Xvalid)

sio.savemat('LinRegression_results.mat',
            mdict={'yHatTrainP':yHatTrainP,'yHatTestP':yHatTestP,
                   'YvalidP':YvalidP,'Ytrain':Ytrain,'Ytest':Ytest})
"""