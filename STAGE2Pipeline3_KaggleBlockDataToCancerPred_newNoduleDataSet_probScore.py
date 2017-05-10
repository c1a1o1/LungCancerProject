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
from keras.models import load_model
from sklearn.metrics import roc_curve
numGivenFeat=4096
numFeats = numGivenFeat*6
numConcatFeats = numGivenFeat*3

trainTestIDs = []
validationIDs = []
trainTestLabels = []
validationLabels = []
stage2IDs = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

with open('stage1_solution.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validationIDs.append(row['id'])
        validationLabels.append(row['cancer'])

with open('stage2_sample_submission.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        stage2IDs.append(row['id'])




def getFeatureDataA(fullDataPath):
    featData = np.load(fullDataPath)
    outVec = np.zeros((1, numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData,                                                   axis=0)  # this causes potential overfit. should remove
    return outVec

def getFeatureData(fileNm,dataFold):
    pathA = os.path.join(dataFold,fileNm)
    return getFeatureDataA(pathA)

dataFolder = '/home/zdestefa/data/KaggleDataBlockInfo2'
dataFolderA = '/home/zdestefa/data/blockFilesResizedVGG19to4096_DanielrkData'
dataFolderB = '/home/zdestefa/data/blockFilesResizedVGG19to4096_highCancer'
resNetFilesAInitial = os.listdir(dataFolderA)
resNetFilesB = os.listdir(dataFolderB)

resNetFilesA = []
for file in resNetFilesAInitial:
    if(file.endswith("label0.npy")):
        resNetFilesA.append(file)

numDataPts = len(resNetFilesA) + len(resNetFilesB)

resNetFiles = []
for file1 in resNetFilesA:
    path1 = os.path.join(dataFolderA,file1)
    resNetFiles.append(path1)
for file2 in resNetFilesB:
    path2 = os.path.join(dataFolderB,file2)
    resNetFiles.append(path2)

x0 = np.zeros((numDataPts,numConcatFeats))
y0 = np.zeros(numDataPts)

y0[0:len(resNetFilesA)]=0
y0[len(resNetFilesA):numDataPts]=1

# for ind in range(numDataPts):
#     if(resNetFilesA[ind].endswith("label0.npy")):
#         y0[ind] = 1
#     else:
#         y0[ind] = 0

numZeros = np.sum(y0<1)
numOne = len(y0)-numZeros

#CHANGE THIS FOR UNIT TESTS

#UNIT TEST VALUES
# numStage2Use = 50
# numTrainUse = 100
# numValidUse = 50
# numPtsUse = 300
# numUseMax = [2*numPtsUse,numPtsUse]

#NORMAL VALUES
numStage2Use = len(stage2IDs)
numTrainUse = len(trainTestIDs)
numValidUse = len(validationIDs)
numUseMax = [numZeros,numOne]

#IMPORTANT: NUMBER OF NODES BEFORE BLOCK PREDICTION
numLayerFeat=16
#numPtsUse = min(numZeros,numOne)
#numUseMax = [2*numPtsUse,numPtsUse]



totalNumPts=np.sum(numUseMax)
x = np.zeros((totalNumPts, numConcatFeats))
y = np.zeros(totalNumPts)


numOut = np.zeros(2)
indsToDrawFrom = np.random.choice(range(len(y0)),size=len(y0))
outInd = 0
for ind0 in indsToDrawFrom:
    curOut = int(y0[ind0])
    if(numOut[curOut] < numUseMax[curOut]):
        numOut[curOut] = numOut[curOut] + 1
        y[outInd] = curOut
        x[outInd,:] = getFeatureDataA(resNetFiles[ind0])
        outInd = outInd+1
        print("Obtained the Neural Net output for block " + str(outInd) + " of " + str(len(y)))

print('Finished getting train/test data for nodule/no-nodule prediction')
print('Num Zero Blocks:' + str(np.sum(y<1)) + ' Num One Block:' + str(np.sum(y>0)))

Yenc = np_utils.to_categorical(y, 2)

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    x, Yenc, random_state=42, stratify=Yenc,test_size=0.20)

input_imgBlocks = Input(shape=(numGivenFeat*3,))
layer1 = Dense(4096, init='normal', activation='relu')(input_imgBlocks)
layer2 = Dense(numLayerFeat, init='normal', activation='sigmoid')(layer1)
outputLayer = Dense(2, init='normal', activation='softmax')(layer2)
noduleModel = Model(input=input_imgBlocks, output=outputLayer)
noduleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("Now fitting Neural Network to predict nodule/no-nodule")
noduleModel.fit(trn_x, trn_y, batch_size=500, nb_epoch=15,
                  verbose=1, validation_data=(val_x, val_y))
noduleModelPreLayer = Model(input=input_imgBlocks,output=layer2)

numRowsTotal = 150
def getFeatDataFromFile2(currentFile):
    initFeatData = np.load(currentFile)
    layerFeatData = noduleModelPreLayer.predict(initFeatData)
    outputData = np.zeros((numRowsTotal,numLayerFeat))
    if(layerFeatData.shape[0] >= numRowsTotal):
        outputData = layerFeatData[0:numRowsTotal,:]
    else:
        for ii in range(numRowsTotal):
            curRind=ii%layerFeatData.shape[0]
            outputData[ii,:] = layerFeatData[curRind,:]
    return outputData

def getFeatDataFromFile3(currentFile):
    initFeatData = np.load(currentFile)
    layerFeatData = noduleModel.predict(initFeatData)
    outputData = np.zeros((numRowsTotal))
    if(layerFeatData.shape[0] >= numRowsTotal):
        outputData = layerFeatData[0:numRowsTotal,1]
    else:
        for ii in range(numRowsTotal):
            curRind=ii%layerFeatData.shape[0]
            outputData[ii] = layerFeatData[curRind,1]
    return outputData


print('Train/Validation Data being obtained from Kaggle')
kaggleFiles = os.listdir(dataFolder)

numFeatsA = numLayerFeat*2
x1 = np.zeros((numTrainUse+numValidUse, 1,numRowsTotal,numLayerFeat))
y1 = np.zeros(numTrainUse+numValidUse)

numZero = 0
numOne = 0
ind = 0

for pInd in range(numTrainUse):
    patID = trainTestIDs[pInd]
    fileName = 'blockInfoOutputMatrix_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x1[ind,0,:,:] = getFeatDataFromFile2(currentFile)
        curL = int(trainTestLabels[pInd])
        y1[ind] = curL
        if(curL<1):
            numZero = numZero + 1
        else:
            numOne = numOne+1
        ind=ind+1
        print("Obtained Kaggle Data for Train/test pt " + str(ind) + " of " + str(numTrainUse))
for pInd in range(numValidUse):
    patID = validationIDs[pInd]
    fileName = 'blockInfoOutputMatrix_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x1[ind,0,:,:] = getFeatDataFromFile2(currentFile)
        curL = int(validationLabels[pInd])
        y1[ind] = curL
        if(curL<1):
            numZero = numZero + 1
        else:
            numOne = numOne+1
        ind=ind+1
        print("Obtained Kaggle Data for validation pt " + str(ind) + " of " + str(numValidUse))


xx=x1
yy=y1
print('Finished getting Kaggle train/test data')
print('Num0: ' + str(np.sum(yy<1)) + '; Num1:' + str(np.sum(yy>0)))

print("Num Data Points" + str(len(yy)))



trn_xx, val_xx, trn_yy2, val_yy2 = cross_validation.train_test_split(xx, yy, random_state=42,
                                                               stratify=yy,
                                                               test_size=0.2)

trn_yy = np_utils.to_categorical(trn_yy2, 2)
val_yy = np_utils.to_categorical(val_yy2, 2)


print('Kaggle Test Data being obtained')
x2 = np.zeros((numStage2Use, 1,numRowsTotal,numLayerFeat))
ind=0
for pInd in range(numStage2Use):
    patID = stage2IDs[pInd]
    fileName = 'blockInfoOutputMatrix_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x2[ind, 0, :,:] = getFeatDataFromFile2(currentFile)
        ind=ind+1
        print("Obtained Kaggle Data for stage2 pt " + str(ind) + " of " + str(numStage2Use))

# input_img2 = Input(shape=(1,1,numRowsTotal))
# actLayer1 = Activation('sigmoid')(input_img2)
# maxLayer2 = MaxPooling2D(pool_size=(1,numRowsTotal))(actLayer1)
# flatten1 = Flatten()(maxLayer2)
# fc1 = Dense(8,init='normal',activation='relu')(flatten1)
# layer2 = Dense(4, init='normal', activation='sigmoid')(fc1)
# outputLayer = Dense(2, init='normal', activation='softmax')(layer2)
# kaggleModel = Model(input=input_img2, output=outputLayer)
# kaggleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# kaggleModel.fit(trn_xx, trn_yy, batch_size=500, nb_epoch=50,
#                   verbose=1, validation_data=(val_xx, val_yy))

input_img2 = Input(shape=(1,numRowsTotal,numLayerFeat))
convLayer1 = Convolution2D(16,5,4,border_mode='valid',activation='relu')(input_img2)
maxLayer2 = MaxPooling2D(pool_size=(10,8))(convLayer1)
flatten1 = Flatten()(maxLayer2)
reluLayer1 = Dense(256,init='normal',activation='relu')(flatten1)
layer2 = Dense(16, init='normal', activation='sigmoid')(reluLayer1)
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
        for ind in range(len(stage2IDs)):
            curPred = pred[ind]
            if (curPred < 0):
                curPred = 0
            writer.writerow({'id': stage2IDs[ind], 'cancer': str(curPred)})

pred = kaggleModel.predict(x2)
prefixString = 'submissions/STAGE2_KaggleNN_NN_Prediction_'
predOut = pred[:,1]
writeKagglePredictionFile(prefixString,predOut)
"""
print('Kaggle Test Data being obtained')
x2 = np.zeros((len(stage2IDs), 1,numRowsTotal,numLayerFeat))
ind=0
for pInd in range(len(stage2IDs)):
    patID = stage2IDs[pInd]
    fileName = 'blockInfoOutputMatrix_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x2[ind, 0, :, :] = getFeatDataFromFile2(currentFile)
        ind=ind+1
        print("Obtained Kaggle Data for stage2 pt " + str(ind) + " of " + str(len(validationIDs)))

input_img2 = Input(shape=(1,numRowsTotal,numLayerFeat))
convLayer1 = Convolution2D(32,8,8,border_mode='valid',activation='relu')(input_img2)
maxLayer2 = MaxPooling2D(pool_size=(4,4))(convLayer1)
flatten1 = Flatten()(maxLayer2)
dropout1 = Dropout(0.25)(flatten1)
fc1 = Dense(2048,init='normal',activation='relu')(dropout1)
layer2 = Dense(256, init='normal', activation='sigmoid')(fc1)
outputLayer = Dense(2, init='normal', activation='softmax')(layer2)
kaggleModel = Model(input=input_img2, output=outputLayer)
kaggleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
kaggleModel.fit(trn_xx, trn_yy, batch_size=500, nb_epoch=50,
                  verbose=1, validation_data=(val_xx, val_yy))
"""