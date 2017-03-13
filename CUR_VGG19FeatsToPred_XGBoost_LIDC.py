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


def train_xgboost():

    print('Train/Validation Data being obtained')
    resNetFiles = os.listdir(dataFolder)
    numDataPts = len(resNetFiles)
    x = np.zeros((numDataPts,numConcatFeats))
    y = np.zeros(numDataPts)

    numZero = 0
    numOne = 0
    indsZero = []
    indsOne = []
    for ind in range(numDataPts):
        x[ind,:] = getFeatureData(resNetFiles[ind])
        if(resNetFiles[ind].endswith("label0.npy")):
            y[ind] = 0
            numZero = numZero + 1
            #indsZero.append(ind)
        else:
            y[ind] = 1
            numOne = numOne+1
            #indsOne.append(ind)

    print('Finished getting train/test data')
    #print('Num0: ' + str(numZero) + '; Num1:' + str(numOne))
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
    return clf


def make_submit():
    clf = train_xgboost()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileNm1 = '/home/zdestefa/data/LIDCxgboostRegressor_' + st + '.npy'
    np.save(fileNm1,clf)
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