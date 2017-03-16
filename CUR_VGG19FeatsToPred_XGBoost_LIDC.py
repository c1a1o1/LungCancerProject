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
    x0 = np.zeros((numDataPts,numConcatFeats))
    y0 = np.zeros(numDataPts)

    for ind in range(numDataPts):
        if(resNetFiles[ind].endswith("label0.npy")):
            y0[ind] = 0
        else:
            y0[ind] = 1

    numZeros = np.sum(y0<1)
    numOne = len(y0)-numZeros
    numPtsUse = min(numZeros,numOne)

    x = np.zeros((2*numPtsUse,numConcatFeats))
    y = np.zeros(2*numPtsUse)

    numOut = np.zeros(2)
    indsToDrawFrom = np.random.choice(range(len(y0)),size=len(y0))
    outInd = 0
    for ind0 in indsToDrawFrom:
        curOut = int(y0[ind0])
        if(numOut[curOut] <= numPtsUse):
            numOut[curOut] = numOut[curOut] + 1
            y[outInd] = curOut
            x[outInd,:] = getFeatureData(resNetFiles[ind0])
            outInd = outInd+1

    print('Finished getting train/test data')
    print('Num Zero Blocks:' + str(np.sum(y<1)) + ' Num One Block:' + str(np.sum(y>0)))

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
    clf.save_model(fileNm1)
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