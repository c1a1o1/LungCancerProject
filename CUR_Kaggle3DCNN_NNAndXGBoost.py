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

#origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

#net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
#net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)

#Trying to model the following network

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

#dataFolderA = '/home/zdestefa/data/blockFilesResizedVGG19to4096'
#resNetFiles = os.listdir(dataFolderA)
#numDataPts = len(resNetFiles)
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
        #currentBlock = huBlocks[nodInd, :, :, :]
        #if K.image_dim_ordering() == 'th':
        #    currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
        #else:
        #    currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
        if (cancerNums[nodInd] > 0):
            YCur = 1
        else:
            YCur = 0
        yAll[allInd] = YCur
        fileInds[allInd] = ind
        noduleInds[allInd] = nodInd
        allInd = allInd+1
        #YUse = np_utils.to_categorical(YCur, 2)
        # print("Ind:" + str(ind))
        #yield (currentBlock.astype('float32'), YUse)

numZeros = np.sum(yAll<1)
numOne = len(yAll)-numZeros
numPtsUse = min(numZeros,numOne)

numUseMax = [2*numPtsUse,numPtsUse]
xUse = np.zeros((3*numPtsUse, 1,64,64,64))
yUse = np.zeros(3*numPtsUse)


numOut = np.zeros(2)
indsToDrawFrom = np.random.choice(range(len(yAll)),size=len(yAll))
outInd = 0
for ind0 in indsToDrawFrom:
    curOut = int(yAll[ind0])
    if(numOut[curOut] < numUseMax[curOut]):
        numOut[curOut] = numOut[curOut] + 1
        yUse[outInd] = curOut
        fileIndUse = int(fileInds[ind0])
        fileP = filesToProcess[fileIndUse]
        curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
        huBlocks = curMATcontent["huBlocksOutput"]
        noduleIndUse = int(noduleInds[ind0])
        currentBlock = huBlocks[noduleIndUse, :, :, :]
        if K.image_dim_ordering() == 'th':
            currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
        else:
            currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)

        xUse[outInd,:,:,:,:] = currentBlock
        outInd = outInd+1
        print("Adding in block " + str(outInd) + " of " + str(3*numPtsUse))

trn_xx, val_xx, trn_yy2, val_yy2 = cross_validation.train_test_split(xUse, yUse, random_state=42,
                                                               stratify=yUse,
                                                               test_size=0.2)

trn_yy = np_utils.to_categorical(trn_yy2, 2)
val_yy = np_utils.to_categorical(val_yy2, 2)


def dataGenerator(filesToProcess,indRange):
    while 1:
        for ind in range(len(indRange)):
            fileP = filesToProcess[indRange[ind]]
            curMATcontent = sio.loadmat(os.path.join(fileFolder, fileP))
            huBlocks = curMATcontent["huBlocksOutput"]
            cancerNums = np.reshape(curMATcontent["outputCancerScores"],len(huBlocks))
            for nodInd in range(len(huBlocks)):
                currentBlock = huBlocks[nodInd, :, :, :]
                if K.image_dim_ordering() == 'th':
                    currentBlock = currentBlock.reshape(1, 1, 64, 64, 64)
                else:
                    currentBlock = currentBlock.reshape(1, 64, 64, 64, 1)
                if(cancerNums[nodInd]>0):
                    YCur=1
                else:
                    YCur=0
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

model.fit(trn_xx,trn_yy,samples_per_epoch = 1000, nb_epoch=10, nb_val_samples=50,
                    verbose=1, validation_data=(val_xx,val_yy))
"""
filesToProcess = os.listdir(fileFolder)
numFiles = len(filesToProcess)
shuffInds = np.random.permutation(numFiles)
endTrainInd = int(np.floor(numFiles*0.8))
trainInds = shuffInds[0:endTrainInd]
validInds = shuffInds[endTrainInd:numFiles]
model.fit_generator(dataGenerator(filesToProcess, trainInds),
                    samples_per_epoch = 1000, nb_epoch=10, nb_val_samples=50,
                    verbose=1, validation_data=dataGenerator(filesToProcess, validInds))

filesToProcess2 = os.listdir(fileFolder2)
numFiles = len(filesToProcess2)
for curInd in range(numFiles):
    print('Obtaining features for file_' + str(curInd) + '_of_' + str(numFiles))
    generateNNoutputFiles(filesToProcess2[curInd])
"""
