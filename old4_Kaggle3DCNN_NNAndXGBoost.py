import numpy as np
import os
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
import cv2
# import mxnet as mx
# import pandas as pd
# from sklearn import cross_validation
# import xgboost as xgb
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

from keras.applications.vgg19 import VGG19
import scipy.io as sio
from scipy.misc import imresize
from sklearn import cross_validation
from scipy.ndimage.interpolation import zoom

import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
from sklearn import random_projection

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


fileFolder2 = '/home/zdestefa/data/huBlockDataSetKaggleOrigSize'
fileFolder = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet'
curDir = '/home/zdestefa/data/rawHUdata'
curDir2 = '/home/zdestefa/data/volResizeInfo'

trainTestIDs = []
validationIDs = []
trainTestLabels = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

with open('stage1_sample_submission.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validationIDs.append(row['id'])

input_shape = (1, 64, 64,64)
model = Sequential()
model.add(Convolution3D(16,9,9,9,border_mode='valid',input_shape=input_shape,activation='relu'))
model.add(MaxPooling3D(pool_size=(3,3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256,init='normal',activation='relu',name='prePredictLayer'))
model.add(Dense(2, init='normal',activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model2 = Model(input=model.input,output=model.get_layer('prePredictLayer').output)

filesToProcess = os.listdir(fileFolder)
numFiles = len(filesToProcess)

numDataPts = 0
for fInd in range(numFiles):
    fileP = filesToProcess[fInd]
    curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
    huBlocks = curMATcontent["huBlocksOutput"]
    print("Init Processing file " + str(fInd) + " of " + str(numFiles))
    numDataPts = numDataPts + len(huBlocks)

xAll = np.zeros((numDataPts,1,64,64,64))
yAll = np.zeros((numDataPts))
allInd = 0
fileInds = np.zeros((numDataPts))
noduleInds = np.zeros((numDataPts))

for fInd in range(numFiles):
    fileP = filesToProcess[fInd]
    curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
    huBlocks = curMATcontent["huBlocksOutput"]
    cancerNums = np.reshape(curMATcontent["outputCancerScores"], len(huBlocks))
    print("Processing file " + str(fInd) + " of " + str(numFiles))
    for nodInd in range(len(huBlocks)):
        if (cancerNums[nodInd] > 0):
            YCur = 1
        else:
            YCur = 0
        yAll[allInd] = YCur
        fileInds[allInd] = fInd
        noduleInds[allInd] = nodInd
        allInd = allInd+1

numZeros = np.sum(yAll<1)
numOne = len(yAll)-numZeros
numPtsUse = min(numZeros,numOne)

numUseMax = [2*numPtsUse,numPtsUse]
xUse = np.zeros((3*numPtsUse, 1,64,64,64))
finalIndsUse = np.zeros(3*numPtsUse)

numOut = np.zeros(2)
indsToDrawFrom = np.random.choice(range(len(yAll)),size=len(yAll))
outInd = 0
for ind0 in indsToDrawFrom:
    curOut = int(yAll[ind0])
    if(numOut[curOut] < numUseMax[curOut]):
        numOut[curOut] = numOut[curOut] + 1
        finalIndsUse[outInd] = ind0
        outInd = outInd+1

numBlocks2 = 3*numPtsUse
shuffInds = np.random.permutation(numBlocks2)
endTrainInd = int(np.floor(numBlocks2*0.8))
trainInds = finalIndsUse[shuffInds[0:endTrainInd]]
validInds = finalIndsUse[shuffInds[endTrainInd:numBlocks2]]

def dataGenerator(filesToProcess,indRange):
    while 1:
        for curI in range(len(indRange)):
            curInd = int(indRange[curI])
            fileIndUse = int(fileInds[curInd])
            fileP = filesToProcess[fileIndUse]
            curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
            huBlocks = curMATcontent["huBlocksOutput"]
            noduleIndUse = int(noduleInds[curInd])
            currentBlock = huBlocks[noduleIndUse, :, :, :]
            if K.image_dim_ordering() == 'th':
                currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
            else:
                currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
            YCur = yAll[curInd]
            YUse = np_utils.to_categorical(YCur, 2)
            yield (currentBlock.astype('float32'),YUse)


def obtainNNoutput(patID):

    curFile = os.path.join(curDir, 'rawDCM_' + patID + '.mat')
    curFile2 = os.path.join(curDir2, 'resizeTuple_' + patID + '.mat')

    matData = sio.loadmat(curFile)
    matData2 = sio.loadmat(curFile2)
    rawHUdata = matData['dcmArrayHU']
    resizeData = np.reshape(matData2['resizeTuple'],3)
    for ii in range(3):
        if(resizeData[ii]<0.1):
            resizeData[ii]=1

    blockDim = 64
    blockDimHalf = 32
    numInXrange = int(np.floor(rawHUdata.shape[0]*resizeData[0]-(blockDimHalf+2)))
    numInYrange = int(np.floor(rawHUdata.shape[1]*resizeData[1]-(blockDimHalf+2)))
    numInZrange = int(np.floor(rawHUdata.shape[2]*resizeData[2]-(blockDimHalf+2)))

    rangeX = range(34, numInXrange, 32)
    rangeY = range(34, numInYrange, 32)
    rangeZ = range(34, numInZrange, 32)

    numGridPts = len(rangeX) * len(rangeY) * len(rangeZ)
    xyzRange = np.meshgrid(rangeX, rangeY, rangeZ)

    xValues = np.reshape(xyzRange[0], numGridPts)
    yValues = np.reshape(xyzRange[1], numGridPts)
    zValues = np.reshape(xyzRange[2], numGridPts)

    print('Matrix Conversion done. Doing Sliding Window...')

    huBlocks = []
    numPossibleLungThreshold = np.floor(64 * 64 * 64 * 0.35)
    resizedHUdata = zoom(rawHUdata,resizeData,mode='nearest',order=0)
    for ii in range(len(xValues)):
        curPtR = xValues[ii]
        curPtC = yValues[ii]
        curPtS = zValues[ii]

        xMin = curPtR - blockDimHalf
        xMax = xMin + blockDim
        yMin = curPtC - blockDimHalf
        yMax = yMin + blockDim
        zMin = curPtS - blockDimHalf
        zMax = zMin + blockDim
        currentHUdataBlock = resizedHUdata[xMin:xMax, yMin:yMax, zMin:zMax]
        numPossibleLung = np.sum(np.logical_and(currentHUdataBlock > -1200, currentHUdataBlock < -700))
        if (numPossibleLung > numPossibleLungThreshold):
            print("Num Lung: " + str(numPossibleLung))
            huBlocks.append(currentHUdataBlock)

    blockPreds = []
    currentInd = 0
    numBlocks = len(huBlocks)
    outputMatrix = np.zeros((numBlocks, 512))
    for nodInd in range(len(huBlocks)):
        currentBlock = huBlocks[nodInd, :, :, :]
        if K.image_dim_ordering() == 'th':
            currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
        else:
            currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
        blockNum = blockNum + 1
        print("obtaining data for block: " + str(blockNum))
        outputMatrix[nodInd,:] = np.reshape(model2.predict(currentBlock),(1,256))
    avgPool = np.mean(outputMatrix, axis=0)
    maxPool = np.mean(outputMatrix, axis=0)
    outVec = np.zeros((1, 512))
    outVec[0, range(256)] = avgPool
    outVec[0, range(256, 512)] = maxPool
    return outVec


model.fit_generator(dataGenerator(filesToProcess, trainInds),
                    samples_per_epoch = 900, nb_epoch=20, nb_val_samples=50,
                    verbose=1, validation_data=dataGenerator(filesToProcess, validInds))



print('Train/Validation Data being obtained from Kaggle')
kaggleFiles = os.listdir(fileFolder2)
numFeatsA = 512
numPossibleDataPts = len(kaggleFiles)
x1 = np.zeros((len(trainTestIDs), numFeatsA))
y1 = np.zeros(len(trainTestIDs))

numZero = 0
numOne = 0
ind = 0
for pInd in range(len(trainTestIDs)):
    patID = trainTestIDs[pInd]
    x1[ind,:] = obtainNNoutput(patID)
    curL = int(trainTestLabels[pInd])
    y1[ind] = curL
    ind=ind+1
    print("Obtained Kaggle Data for pt " + str(ind) + " of " + str(len(trainTestIDs)))


trn_xx, val_xx, trn_yy2, val_yy2 = cross_validation.train_test_split(x1, y1, random_state=42,
                                                               stratify=y1,
                                                               test_size=0.2)

trn_yy = np_utils.to_categorical(trn_yy2, 2)
val_yy = np_utils.to_categorical(val_yy2, 2)

print('Kaggle Test Data being obtained')
x2 = np.zeros((len(validationIDs), numFeatsA))
ind=0
for pInd in range(len(validationIDs)):
    patID = validationIDs[pInd]
    x2[ind, :] = obtainNNoutput(patID)
    ind = ind + 1
    print("Obtained Kaggle Data for pt " + str(ind) + " of " + str(len(trainTestIDs)))


input_img2 = Input(shape=(512,))
layer2 = Dense(32, init='normal', activation='sigmoid')(input_img2)
outputLayer = Dense(2, init='normal', activation='softmax')(layer2)
kaggleModel = Model(input=input_img2, output=outputLayer)
kaggleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
kaggleModel.fit(trn_xx, trn_yy, batch_size=500, nb_epoch=50,
                  verbose=1, validation_data=(val_xx, val_yy))

def writeKagglePredictionFile(prefixString,pred):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileName = prefixString + st + '.csv'

    with open(fileName, 'w') as csvfile:
        fieldnames = ['id', 'cancer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ind in range(len(validationIDs)):
            curPred = pred[ind]
            if (curPred < 0):
                curPred = 0
            writer.writerow({'id': validationIDs[ind], 'cancer': str(curPred)})

pred = kaggleModel.predict(x2)
prefixString = 'submissions/Kaggle3DCNN_NN_Prediction_'
predOut = pred[:,1]
writeKagglePredictionFile(prefixString,predOut)

clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

clf.fit(trn_xx, trn_yy2, eval_set=[(val_xx, val_yy2)], verbose=True,
        eval_metric='logloss', early_stopping_rounds=100)

pred2 = clf.predict(x2)
prefixString2 = 'submissions/Kaggle3DCNN_XGBoost_Prediction_'
writeKagglePredictionFile(prefixString2,pred2)

