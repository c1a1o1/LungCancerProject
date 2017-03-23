import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import scipy.io as sio

numGivenFeat=4096
numFeats = 75

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

dataFolder = '/home/zdestefa/data/KaggleXGBoostPreds4'

def getFeatureData(featData):
    featData2 = np.reshape(featData,featData.size)
    if(featData.size<numFeats):
        outputData= np.random.choice(featData2,size=(1,numFeats),replace=True)
    else:
        outputData =  np.random.choice(featData2,size=(1,numFeats),replace=False)
    finalOutput = np.sort(outputData)
    return finalOutput[::-1]





def train_xgboost():

    print('Train/Validation Data being obtained')
    resNetFiles = os.listdir(dataFolder)
    numPossibleDataPts = len(resNetFiles)
    x0 = np.zeros((numPossibleDataPts, numFeats))
    y0 = np.zeros(numPossibleDataPts)

    numZero = 0
    numOne = 0
    ind = 0
    for pInd in range(len(trainTestIDs)):
        patID = trainTestIDs[pInd]
        fileName = 'xgBoostPreds_'+patID+'.npy'
        currentFile = os.path.join(dataFolder, fileName)
        if(os.path.isfile(currentFile)):
            initFeatData = np.load(currentFile)
            x0[ind,:] = getFeatureData(initFeatData)
            curL = int(trainTestLabels[pInd])
            y0[ind] = curL
            ind=ind+1

    trn_x0, val_x, trn_y0, val_y = cross_validation.train_test_split(
        x0, y0, random_state=42, stratify=y0,test_size=0.2)

    numZeros = np.sum(trn_y0 < 1)
    numOne = len(trn_y0) - numZeros
    numPtsUse = min(numZeros, numOne)
    # numPtsUse = 300

    numUseMax = [numPtsUse, numPtsUse]
    x = np.zeros((2 * numPtsUse, numFeats))
    y = np.zeros(2 * numPtsUse)


    #THIS SETUP PROVED TO BE THE WORST
    #numUseMax = [numZero,numOne] #use all the data
    #x = np.zeros((len(trainTestIDs),numFeats))
    #y = np.zeros(len(trainTestIDs))

    numCat = np.zeros(2)
    indsUse2 = np.random.choice(range(len(trn_y0)),len(trn_y0)) #randomized order
    cInd = 0
    for pInd in indsUse2:
        curLabel = int(trn_y0[pInd])
        if(numCat[curLabel]<numUseMax[curLabel]):
            numCat[curLabel] = numCat[curLabel]+1
            x[cInd,:] = trn_x0[pInd,:]
            y[cInd] = curLabel
            cInd = cInd+1

    print('Finished getting train/test data')
    print('Num0: ' + str(np.sum(y<1)) + '; Num1:' + str(np.sum(y>0)))

    print("Num Data Points" + str(len(y)))

    #trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
    #                                                               test_size=0.2)

    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(x, y, eval_set=[(val_x, val_y)], verbose=True,
            eval_metric='logloss', early_stopping_rounds=100)

    yhat = clf.predict(x)
    yhatVal = clf.predict(val_x)

    sio.savemat('/home/zdestefa/data/PatientDataROCcurveBalancedSplit.mat',
                {"yhat": yhat, "y": y, "yhatVal": yhatVal, "val_y": val_y})

    return clf


def make_submit():
    clf = train_xgboost()



if __name__ == '__main__':
    #calc_featuresA()
    make_submit()