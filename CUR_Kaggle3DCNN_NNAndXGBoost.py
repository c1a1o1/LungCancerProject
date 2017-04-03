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
fileFolder2 = '/home/zdestefa/data/huBlockDataSetKaggleOrigSize'
fileFolder = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet'

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
for ind in range(numFiles):
    fileP = filesToProcess[ind]
    curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
    huBlocks = curMATcontent["huBlocksOutput"]
    print("Init Processing file " + str(ind) + " of " + str(numFiles))
    numDataPts = numDataPts + len(huBlocks)

xAll = np.zeros((numDataPts,1,64,64,64))
yAll = np.zeros((numDataPts))
allInd = 0
fileInds = np.zeros(numDataPts)
noduleInds = np.zeros(numDataPts)

for ind in range(numFiles):
    fileP = filesToProcess[ind]
    curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
    huBlocks = curMATcontent["huBlocksOutput"]
    cancerNums = np.reshape(curMATcontent["outputCancerScores"], len(huBlocks))
    print("Processing file " + str(ind) + " of " + str(numFiles))
    for nodInd in range(len(huBlocks)):
        if (cancerNums[nodInd] > 0):
            YCur = 1
        else:
            YCur = 0
        yAll[allInd] = YCur
        fileInds[allInd] = ind
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
        print("Adding in block " + str(outInd) + " of " + str(3*numPtsUse))

numBlocks2 = 3*numPtsUse
shuffInds = np.random.permutation(numBlocks2)
endTrainInd = int(np.floor(numBlocks2*0.8))
trainInds = finalIndsUse[shuffInds[0:endTrainInd]]
validInds = finalIndsUse[shuffInds[endTrainInd:numFiles]]

def dataGenerator(filesToProcess,indRange):
    while 1:
        for ind in indRange:
            fileIndUse = int(fileInds[ind])
            fileP = filesToProcess[fileIndUse]
            curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
            huBlocks = curMATcontent["huBlocksOutput"]
            noduleIndUse = int(noduleInds[ind])
            currentBlock = huBlocks[noduleIndUse, :, :, :]
            if K.image_dim_ordering() == 'th':
                currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
            else:
                currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
            YCur = yAll[ind]
            YUse = np_utils.to_categorical(YCur, 2)
            #print("Ind:" + str(ind))
            yield (currentBlock.astype('float32'),YUse)




def generateNNoutputFiles(fileP):
    patID = fileP[9:len(fileP) - 4]
    curMATcontent = sio.loadmat(os.path.join(fileFolder2, fileP))
    huBlocks = curMATcontent["huBlocksOutput"]
    rPrefix = 'resnetFeats_'
    folder4096prefix = 'data/blockFilesResizedUserNNPrediction/'
    folder4096prefix2 = 'data/blockFilesResizedUserNNLastLayer/'
    blockNum = 0
    for nodInd in range(len(huBlocks)):
        currentBlock = huBlocks[nodInd, :, :, :]
        if K.image_dim_ordering() == 'th':
            currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
        else:
            currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
        noduleSuffix = '_Block_' + str(blockNum)
        blockNum = blockNum + 1
        fileName3 = folder4096prefix + rPrefix + patID + noduleSuffix
        fileName4 = folder4096prefix2 + rPrefix + patID + noduleSuffix
        print("obtaining resnet data for block: " + str(blockNum))
        output3 = model.predict(currentBlock)
        output4 = model2.predict(currentBlock)
        np.save(fileName3, output3)
        np.save(fileName4, output4)


model.fit_generator(dataGenerator(filesToProcess, trainInds),
                    samples_per_epoch = 1000, nb_epoch=10, nb_val_samples=50,
                    verbose=1, validation_data=dataGenerator(filesToProcess, validInds))
"""
filesToProcess2 = os.listdir(fileFolder2)
numFiles = len(filesToProcess2)
for curInd in range(numFiles):
    print('Obtaining features for file_' + str(curInd) + '_of_' + str(numFiles))
    generateNNoutputFiles(filesToProcess2[curInd])
"""
