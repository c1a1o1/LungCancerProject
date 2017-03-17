import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os

numGivenFeat=4096
numFeats = 50

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

dataFolder = '/home/zdestefa/data/KaggleXGBoostPreds2'

def getFeatureData(featData):
    outputData = np.zeros((1,numFeats))
    outputData[0, range(featData.size)] = np.reshape(featData, (1, featData.size))
    return outputData




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
            if(initFeatData.size >= 50): #TEMP WORKAROUND. TODO: FIX THIS
                x0[ind,:] = getFeatureData(initFeatData)
                curL = int(trainTestLabels[pInd])
                y0[ind] = curL
                if(curL<1):
                    numZero = numZero + 1
                else:
                    numOne = numOne+1
                ind=ind+1

    numberUse = min(numOne,numZero)
    x = np.zeros((2*numberUse, numFeats))
    y = np.zeros(2*numberUse)
    numCat = np.zeros(2)
    indsUse2 = np.random.choice(range(len(y0)),len(y0)) #randomized order
    cInd = 0
    for pInd in indsUse2:
        curLabel = int(y0[pInd])
        if(numCat[curLabel]<numberUse):
            numCat[curLabel] = numCat[curLabel]+1
            x[cInd,:] = x0[pInd,:]
            y[cInd] = curLabel
            cInd = cInd+1

    print('Finished getting train/test data')
    print('Num0: ' + str(np.sum(y<1)) + '; Num1:' + str(np.sum(y>0)))

    print("Num Data Points" + str(len(y)))

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.2)

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
    return clf


def make_submit():
    clf = train_xgboost()



    print('Kaggle Test Data being obtained')
    x2 = np.zeros((len(validationIDs), numFeats))
    ind = 0
    for id in validationIDs:
        fileName = 'xgBoostPreds_' + id + '.npy'
        currentFile = os.path.join(dataFolder, fileName)
        if (os.path.isfile(currentFile)):
            initFeatData = np.load(currentFile)
        x2[ind, :] = getFeatureData(initFeatData)
        ind = ind + 1

    print('Finished getting kaggle test data')

    pred = clf.predict(x2)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileName = 'submissions/BlockDataSubmission_' + st + '.csv'

    with open(fileName, 'w') as csvfile:
        fieldnames = ['id', 'cancer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ind in range(len(validationIDs)):
            writer.writerow({'id': validationIDs[ind], 'cancer': str(pred[ind])})



if __name__ == '__main__':
    #calc_featuresA()
    make_submit()