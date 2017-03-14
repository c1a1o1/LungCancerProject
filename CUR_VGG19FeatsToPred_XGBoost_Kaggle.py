import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096Kaggle'

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

    file1 = '/home/zdestefa/data/LIDCxgboostRegressor_2017_03_13__03_52_04.npy'
    clf = np.load(file1)
    for ind in range(numDataPts):
        probs = clf.predict(getFeatureData(resNetFiles[ind]))
        print(probs)



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