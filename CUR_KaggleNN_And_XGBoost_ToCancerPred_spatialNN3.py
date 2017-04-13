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
from keras.models import load_model

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


numGivenFeat=4096
numFeats = numGivenFeat*6
numLayerFeat=256
numConcatFeats = numGivenFeat*3

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

def getFeatureData(fileNm,dataFold):
    featData = np.load(os.path.join(dataFold,fileNm))
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData, axis=0)  # this causes potential overfit. should remove
    return outVec

dataFolder = '/home/zdestefa/data/KaggleDataBlockInfo3'
dataFolderA = '/home/zdestefa/data/blockFilesResizedVGG19to4096'
resNetFiles = os.listdir(dataFolderA)
numDataPts = len(resNetFiles)


x0 = np.zeros((numDataPts,numConcatFeats))
y0 = np.zeros(numDataPts)

for ind in range(numDataPts):
    if(resNetFiles[ind].endswith("label0.npy")):
        y0[ind] = 1
    else:
        y0[ind] = 0



numZeros = np.sum(y0<1)
numOne = len(y0)-numZeros
#numPtsUse = min(numZeros,numOne)
numPtsUse = 2000

numUseMax = [2*numPtsUse,numPtsUse]
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
        x[outInd,:] = getFeatureData(resNetFiles[ind0],dataFolderA)
        outInd = outInd+1
        print("Obtained the Neural Net output for block " + str(outInd) + " of " + str(len(y)))

print('Finished getting train/test data for nodule/no-nodule prediction')
print('Num Zero Blocks:' + str(np.sum(y<1)) + ' Num One Block:' + str(np.sum(y>0)))

Yenc = np_utils.to_categorical(y, 2)

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    x, Yenc, random_state=42, stratify=Yenc,test_size=0.20)

input_imgBlocks = Input(shape=(numGivenFeat*3,))
layer1 = Dense(4096, init='normal', activation='relu')(input_imgBlocks)
layer2 = Dense(256, init='normal', activation='sigmoid')(layer1)
outputLayer = Dense(2, init='normal', activation='softmax')(layer2)
noduleModel = Model(input=input_imgBlocks, output=outputLayer)
noduleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("Now fitting Neural Network to predict nodule/no-nodule")
noduleModel.fit(trn_x, trn_y, batch_size=500, nb_epoch=30,
                  verbose=1, validation_data=(val_x, val_y))
noduleModelPreLayer = Model(input=input_imgBlocks,output=layer2)


numRowsTotal = 150
finalShape = [16,16,16,256]
def getFeatDataFromFile2(currentFile):
    initFeatData = np.load(currentFile)
    outputTensor = np.zeros(finalShape)
    for ii in range(initFeatData.shape[0]):
        for jj in range(initFeatData.shape[1]):
            for kk in range(initFeatData.shape[2]):
                inputData = initFeatData[ii,jj,kk,:]
                inputVector = np.reshape(inputData,(1,len(inputData)))
                vectData = noduleModelPreLayer.predict(inputVector)
                outputTensor[ii,jj,kk,:] = vectData
    finalOutput = np.mean(outputTensor,axis=2)


    return finalOutput



print('Train/Validation Data being obtained from Kaggle')
kaggleFiles = os.listdir(dataFolder)
numFeatsA = numLayerFeat*2
x1 = np.zeros((len(trainTestIDs), 1,16,16,256))
y1 = np.zeros(len(trainTestIDs))

numZero = 0
numOne = 0
ind = 0
for pInd in range(len(trainTestIDs)):
    patID = trainTestIDs[pInd]
    fileName = 'blockInfoOutput4DTensor_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x1[ind,0,:,:,:] = getFeatDataFromFile2(currentFile)
        curL = int(trainTestLabels[pInd])
        y1[ind] = curL
        if(curL<1):
            numZero = numZero + 1
        else:
            numOne = numOne+1
        ind=ind+1
        print("Obtained Kaggle Data for pt " + str(ind) + " of " + str(len(trainTestIDs)))

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
x2 = np.zeros((len(validationIDs), 1,16,16,256))
ind=0
for pInd in range(len(validationIDs)):
    patID = validationIDs[pInd]
    fileName = 'blockInfoOutput4DTensor_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        x2[ind, 0, :, :, :] = getFeatDataFromFile2(currentFile)
        ind=ind+1
        print("Obtained Kaggle Data for pt " + str(ind) + " of " + str(len(validationIDs)))


input_img2 = Input(shape=(1,16,16,256))
convLayer1 = Convolution3D(32,4,4,1,border_mode='valid',activation='relu')(input_img2)
maxLayer2 = MaxPooling3D(pool_size=(4,4,1))(convLayer1)
dropout1 = Dropout(0.25)(maxLayer2)
flatten1 = Flatten()(dropout1)
fc1 = Dense(2048,init='normal',activation='relu')(flatten1)
fc2 = Dense(1024,init='normal',activation='relu')(fc1)
fc3 = Dense(256,init='normal',activation='relu')(fc2)
layer2 = Dense(64, init='normal', activation='sigmoid')(fc3)
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
prefixString = 'submissions/KaggleNN_NN_Prediction_'
predOut = pred[:,1]
writeKagglePredictionFile(prefixString,predOut)
"""
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
prefixString2 = 'submissions/KaggleNN_XGBoost_Prediction_'
writeKagglePredictionFile(prefixString2,pred2)
"""