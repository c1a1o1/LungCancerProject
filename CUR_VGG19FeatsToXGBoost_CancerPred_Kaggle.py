import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import csv

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096'
dataFolder2 = '/home/zdestefa/data/blockFilesResizedVGG19to4096Kaggle'

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


def train_xgboost():

    print('Train/Validation Data being obtained')
    resNetFiles = os.listdir(dataFolder)
    numDataPts = len(resNetFiles)
    x0 = np.zeros((numDataPts,numConcatFeats))
    y0 = np.zeros(numDataPts)

    numZero = 0
    numOne = 0
    indsZero = []
    indsOne = []
    for ind in range(numDataPts):
        x0[ind,:] = getFeatureData(resNetFiles[ind],dataFolder)
        if(resNetFiles[ind].endswith("label0.npy")):
            y0[ind] = 0
            numZero = numZero + 1
            #indsZero.append(ind)
        else:
            y0[ind] = 1
            numOne = numOne+1
            #indsOne.append(ind)

    #indsUse = np.random.choice(range(len(y0)),size=300)
    x = x0#[indsUse,:]
    y = y0#[indsUse]

    print('Finished getting train/test data')
    print('Num0: ' + str(numZero) + '; Num1:' + str(numOne))
    #numUse = min(numZero,numOne)

    # x2 = np.zeros((numUse*2,numConcatFeats))
    # y2 = np.zeros(numUse*2)
    # zerosIndsUse = np.random.choice(indsZero,numUse,replace=False)
    # oneIndsUse = np.random.choice(indsOne,numUse,replace=False)
    #
    # x2[0:numUse,:] = x[zerosIndsUse,:]
    # y2[0:numUse] = y[zerosIndsUse]
    # x2[numUse:numUse*2, :] = x[oneIndsUse, :]
    # y2[numUse:numUse*2] = y[oneIndsUse]

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

    resNetFiles2 = os.listdir(dataFolder2)
    numDataPts2 = len(resNetFiles2)
    x1 = np.zeros((numDataPts2, numConcatFeats))

    saveFolder = '/home/zdestefa/data/KaggleXGBoostPreds'

    for patID in trainTestIDs:
        blockFiles = [file1 for file1 in resNetFiles2 if (patID in file1)]
        if(len(blockFiles)>0):
            blockPreds = []
            for fileA in blockFiles:
                featD = getFeatureData(fileA, dataFolder2)
                blockPreds.append(clf.predict(featD))
            saveFile = os.path.join(saveFolder,'xgBoostPreds_'+patID+'.npy')
            np.save(saveFile,blockPreds)


    return clf


def make_submit():
    clf = train_xgboost()
    # ts = time.time()
    # st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    # fileNm1 = '/home/zdestefa/data/LIDCxgboostRegressor_' + st + '.npy'
    # clf.save_model(fileNm1)
    #np.save(fileNm1,clf)
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