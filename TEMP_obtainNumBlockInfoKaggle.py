import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import scipy.io as sio

numGivenFeat=4096
numFeats = 150

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

dataFolder = '/home/zdestefa/data/KaggleXGBoostPreds3'



print('Train/Validation Data being obtained')
resNetFiles = os.listdir(dataFolder)

numBlocksArray = []
numZero = 0
numOne = 0

for pInd in range(len(trainTestIDs)):
    patID = trainTestIDs[pInd]
    fileName = 'xgBoostPreds_'+patID+'.npy'
    currentFile = os.path.join(dataFolder, fileName)
    if(os.path.isfile(currentFile)):
        initFeatData = np.load(currentFile)
        numBlocksArray.append(initFeatData.size)
        if(int(trainTestLabels[pInd])>0):
            numOne = numOne+1
        else:
            numZero=numZero+1
        print(initFeatData.size)

print("Num Zero: " + str(numZero))
print("Num One: " + str(numOne))

saveFile = '/home/zdestefa/data/sizeInfoKaggle.mat'
sio.savemat(saveFile, {"numBlocksArray":numBlocksArray})


