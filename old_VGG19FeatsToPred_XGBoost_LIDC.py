import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096'

def getFeatureData(fileNm):
    featData = np.load(os.path.join(dataFolder,fileNm))
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData, axis=0)  # this causes potential overfit. should remove
    return outVec

def getBlockInfoData(resNetFileNm):
    blockDataSuffix = resNetFileNm[12:len(resNetFileNm)]
    blockDataFileNm = 'blockData_'+blockDataSuffix
    blockDataFile = os.path.join(dataFolder,blockDataFileNm)
    infoArray = np.load(blockDataFile)
    cancerScore = infoArray[0]
    noduleRadius = infoArray[1]
    lungPixelNum = infoArray[2]
    return cancerScore,noduleRadius,lungPixelNum


def train_xgboost():

    print('Train/Validation Data being obtained')
    files = os.listdir(dataFolder)

    resNetFiles = [file1 for file1 in files if file1.startswith("resnetFeats")]
    cancerScores = np.zeros(len(resNetFiles))
    nodRadii = np.zeros(len(resNetFiles))
    lungNums = np.zeros(len(resNetFiles))
    for fInd in range(len(resNetFiles)):
        cancerScores[fInd],nodRadii[fInd],lungNums[fInd] = getBlockInfoData(resNetFiles[fInd])

    cancerPts = np.where(cancerScores>2)
    cancerPtsUse = cancerPts

    noCancerPts = np.where(cancerScores<1)
    lungBlocks = np.where(lungNums>(64*64*64*0.05))
    otherBlocksInclude = [ptNum for ptNum in noCancerPts if ptNum in lungBlocks]
    otherPtsUse = otherBlocksInclude

    numCancer = 500
    numOther = 500
    if(numCancer < len(cancerPts)):
        cancerInds = np.random.choice(range(len(cancerPts)), size=numCancer, replace=False)
        cancerPtsUse = cancerPts[cancerInds]
    if(numOther < len(otherBlocksInclude)):
        otherInds = np.random.choice(range(len(otherBlocksInclude)),size=numOther,replace=False)
        otherPtsUse = otherBlocksInclude[otherInds]

    numDataPts = len(cancerPtsUse)+len(otherPtsUse)
    x = np.zeros((numDataPts,numConcatFeats))
    y = np.zeros(numDataPts)

    for ind in range(len(cancerPtsUse)):
        x[ind,:] = getFeatureData(resNetFiles[cancerPtsUse[ind]])
        y[ind] = 1
    for ind in range(len(otherPtsUse)):
        x[ind+len(cancerPtsUse),:] = getFeatureData(resNetFiles[otherPtsUse[ind]])
        y[ind+len(cancerPtsUse)] = 0

    print('Finished getting train/test data')
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
    return clf


def make_submit():
    clf = train_xgboost()
    """
    print('Kaggle Test Data being obtained')
    x2 = np.zeros((len(validationIDs), numConcatFeats))
    ind = 0
    for id in validationIDs:
        x2[ind, :] = getFeatureData(id)
        ind = ind + 1
    print('Finished getting kaggle test data')

    pred = clf.predict(x2)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileName = 'submissions/VGG19PlusXGBoost_' + st + '.csv'

    with open(fileName, 'w') as csvfile:
        fieldnames = ['id', 'cancer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ind in range(len(validationIDs)):
            writer.writerow({'id': validationIDs[ind], 'cancer': str(pred[ind])})
    """


if __name__ == '__main__':
    #calc_featuresA()
    make_submit()