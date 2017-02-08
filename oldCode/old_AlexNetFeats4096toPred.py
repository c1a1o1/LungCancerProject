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
import csv
from sklearn.decomposition import PCA,KernelPCA
from convnetskeras.convnets import preprocess_image_batch, convnet
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

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids


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
numObs = 1595
allXData = np.zeros((numObs*100,4096))
index = 0

for kk in range(len(trainTestIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+trainTestIDs[kk]+'.npy'
    curX = np.load(curFile)
    startInd = index*100
    endInd = (index+1)*100
    allXData[startInd:endInd,:] = curX
    index = index+1

for kk in range(len(validationIDs)):
    curFile = 'AlexNetFeatures2D/feats3D_conv2_'+validationIDs[kk]+'.npy'
    curX = np.load(curFile)
    startInd = index*100
    endInd = (index+1)*100
    allXData[startInd:endInd,:] = curX
    index = index+1

transformer = KernelPCA(n_components=20)
#transformer = random_projection.GaussianRandomProjection()
allXnew = transformer.fit_transform(allXData)
print(allXnew.shape)

allPats = np.zeros((1595,20*100))
for kk in range(len(trainTestIDs)+len(validationIDs)):
    startInd = kk * 100
    endInd = (kk + 1) * 100
    curPat = allXnew[startInd:endInd,:]
    allPats[kk,:] = np.reshape(curPat,20*100)

allPatsNew = transformer.fit_transform(allPats)
print(allPatsNew.shape)

Xdata = allPatsNew[0:numTrainTest,:]
Xvalid = allPatsNew[numTrainTest:(numTrainTest+numValid),:]

print(Xdata.shape)
print(Xvalid.shape)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xdata,trainTestLabels,test_size=0.1,random_state=42)

clf = RandomForestClassifier(max_depth=8, n_estimators=20, max_features=10)
clf = clf.fit(Xtrain,Ytrain)

yHatTrainP = clf.predict_proba(Xtrain)
yHatTestP = clf.predict_proba(Xtest)
YvalidP = clf.predict_proba(Xvalid)

sio.savemat('DualPCA_RF_results.mat',
            mdict={'yHatTrainP':yHatTrainP,'yHatTestP':yHatTestP,
                   'YvalidP':YvalidP,'Ytrain':Ytrain,'Ytest':Ytest})
