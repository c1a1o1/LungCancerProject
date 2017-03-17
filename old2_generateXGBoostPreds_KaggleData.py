import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom
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
import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import csv

from keras.applications.vgg19 import VGG19
import scipy.io as sio
from scipy.misc import imresize
numGivenFeat=4096
numConcatFeats = numGivenFeat*3


dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096'
dataFolder2 = '/home/zdestefa/data/blockFilesResizedVGG19to4096Kaggle'


curDir = '/home/zdestefa/data/rawHUdata'

print('Loading Binary Array')

trainTestIDs = []
trainTestLabels = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])


def getFeatureData(fileNm,dataFold):
    featData = np.load(os.path.join(dataFold,fileNm))
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData, axis=0)  # this causes potential overfit. should remove
    return outVec

matFiles = os.listdir(curDir)

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

#net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)


def getResNetData(curData):
    curImg = curData
    #curImg[curImg==-2000]=0
    batch = []
    #for i in range(0, curData.shape[0] - 3, 3):
    for i in range(curData.shape[2]):
        tmp = []
        for j in range(3):
            img = curImg[i]
            #
            #NOTE: DO NOT DO THE EQUALIZEHIST PROCEDURE
            #   RESULTS ARE DRAMATICALLY BETTER WITHOUT IT
            #   WENT FROM 0.52 LOG LOSS TO 0.33 LOG LOSS
            #
            #RESULTS ARE ALSO MUCH BETTER REPLACIATING THE IMAGE
            #   IN EACH CHANNEL RATHER THAN TRY TO COMBINE
            #   IMAGES IN THE COLOR CHANNELS
            #
            #img = 255.0 / np.amax(img) * img
            #img = cv2.equalizeHist(img.astype(np.uint8))
            #img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    feats3 = net3.predict(batch)
    return feats3



print('Train/Validation Data being obtained')
resNetFiles = os.listdir(dataFolder)
numDataPts = len(resNetFiles)
x0 = np.zeros((numDataPts,numConcatFeats))
y0 = np.zeros(numDataPts)

for ind in range(numDataPts):
    if(resNetFiles[ind].endswith("label0.npy")):
        y0[ind] = 0
    else:
        y0[ind] = 1

numZeros = np.sum(y0<1)
numOne = len(y0)-numZeros
numPtsUse = min(numZeros,numOne)

x = np.zeros((2*numPtsUse,numConcatFeats))
y = np.zeros(2*numPtsUse)

numOut = np.zeros(2)
indsToDrawFrom = np.random.choice(range(len(y0)),size=len(y0))
outInd = 0
for ind0 in indsToDrawFrom:
    curOut = int(y0[ind0])
    if(numOut[curOut] <= numPtsUse):
        numOut[curOut] = numOut[curOut] + 1
        y[outInd] = curOut
        x[outInd,:] = getFeatureData(resNetFiles[ind0])
        outInd = outInd+1

print('Finished getting train/test data')
print('Num Zero Blocks:' + str(np.sum(y<1)) + ' Num One Block:' + str(np.sum(y>0)))

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                               test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                       n_estimators=1500,
                       min_child_weight=9,
                       learning_rate=0.05,
                       nthread=8,
                       subsample=0.80,
                       colsample_bytree=0.80,
                       seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True,
        eval_metric='logloss', early_stopping_rounds=100)

for fInd in range(len(matFiles)):
    print('Now Processing File ' + str(fInd) + ' of ' + str(len(matFiles)))

    curFile = os.path.join(curDir,matFiles[fInd])
    patID = matFiles[fInd][7:len(matFiles[fInd])-4]

    #huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    #resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    #huDataFileNm = huDataFilePrefix+patID+'.npy'
    #resizeFileNm = resizeTupleFilePrefix+patID+'.npy'

    matData = sio.loadmat(curFile)
    rawHUdata = matData['dcmArrayHU']

    blockDim = 64
    blockDimHalf = 32
    maxX = rawHUdata.shape[0]-(blockDim+2)
    maxY = rawHUdata.shape[1]-(blockDim+2)
    maxZ = rawHUdata.shape[2]-(blockDim+2)
    rangeX = range(34,maxX,32)
    rangeY = range(34,maxY,32)
    rangeZ = range(34, maxZ, 32)
    numGridPts = len(rangeX)*len(rangeY)*len(rangeZ)
    xyzRange = np.meshgrid(rangeX,rangeY,rangeZ)

    xValues = np.reshape(xyzRange[0],numGridPts)
    yValues = np.reshape(xyzRange[1],numGridPts)
    zValues = np.reshape(xyzRange[2],numGridPts)

    print('Matrix Conversion done. Doing Sliding Window...')

    huBlocks = []
    numPossibleLungThreshold = np.floor(64 * 64 * 64 * 0.35)
    for ii in range(len(xValues)):
        curPtR = xValues[ii]
        curPtC = yValues[ii]
        curPtS = zValues[ii]

        xMin = curPtR-blockDimHalf
        xMax = xMin + blockDim
        yMin = curPtC-blockDimHalf
        yMax = yMin + blockDim
        zMin = curPtS-blockDimHalf
        zMax = zMin + blockDim
        currentHUdataBlock = rawHUdata[xMin:xMax,yMin:yMax,zMin:zMax]
        numPossibleLung = np.sum(np.logical_and(currentHUdataBlock > -1200, currentHUdataBlock < -700))
        if(numPossibleLung>numPossibleLungThreshold ):
            print("Num Lung: " + str(numPossibleLung))
            huBlocks.append(currentHUdataBlock)

    saveFolder = '/home/zdestefa/data/KaggleXGBoostPreds'

    print("Now putting each lung block through ResNet and XGBoost")

    blockPreds = []
    displayInd = 1
    for block in huBlocks:
        print("Processing block " + str(displayInd) + " of " + str(len(huBlocks)))
        displayInd = displayInd+1
        feats = getResNetData(block)
        outVec = np.zeros((1, numConcatFeats))
        outVec[0, 0:numGivenFeat] = np.mean(feats, axis=0)
        outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(feats,axis=0)  # this causes potential overfit. should remove
        outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(feats,axis=0)  # this causes potential overfit. should remove
        blockPreds.append(clf.predict(outVec))

    print("Now outputting XGBoost Prediction Array")
    saveFile = os.path.join(saveFolder, 'xgBoostPreds_' + patID + '.npy')
    np.save(saveFile, blockPreds)

    # huBlocksOutput = np.zeros((len(huBlocks),64,64,64))
    # curI = 0
    # for block in huBlocks:
    #     huBlocksOutput[curI,:,:,:]=block
    #     curI = curI + 1
    # sio.savemat(outFile, {"huBlocksOutput": huBlocksOutput})
