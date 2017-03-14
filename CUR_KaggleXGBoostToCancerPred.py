import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os

numGivenFeat=4096
numConcatFeats = 4

trainTestIDs = []
trainTestLabels = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

dataFolder = '/home/zdestefa/data/KaggleXGBoostPreds'

def getFeatureData(fileData):
    featData = np.load(fileData)
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0] = np.mean(featData)
    outVec[0, 1] = np.max(featData)  # this causes potential overfit. should remove
    outVec[0, 2] = np.min(featData)  # this causes potential overfit. should remove
    outVec[0, 3] = np.std(featData)  # this causes potential overfit. should remove
    return outVec


def train_xgboost():

    print('Train/Validation Data being obtained')
    resNetFiles = os.listdir(dataFolder)
    numPossibleDataPts = len(resNetFiles)
    x0 = np.zeros((numPossibleDataPts,numConcatFeats))
    y0 = np.zeros(numPossibleDataPts)

    numZero = 0
    numOne = 0
    ind = 0
    for pInd in range(len(trainTestIDs)):
        patID = trainTestIDs[pInd]
        fileName = 'xgBoostPreds_'+patID+'.npy'
        currentFile = os.path.join(dataFolder, fileName)
        if(os.path.isfile(currentFile)):
            x0[ind,:] = getFeatureData(currentFile)
            curL = int(trainTestLabels[pInd])
            y0[ind] = curL
            if(curL<1):
                numZero = numZero + 1
            else:
                numOne = numOne+1
            ind=ind+1

    x = x0[0:ind,:]
    y = y0[0:ind]
    print('Finished getting train/test data')
    print('Num0: ' + str(numZero) + '; Num1:' + str(numOne))

    print("Num Data Points" + str(len(y)))

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.1)

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