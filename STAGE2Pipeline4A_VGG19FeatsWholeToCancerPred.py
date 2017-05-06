import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

trainTestIDs = []
trainTestLabels = []
validationIDs = []
validationLabels = []
stage2IDs = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

with open('stage1_solution.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validationIDs.append(row['id'])
        validationLabels.append(row['cancer'])

with open('stage2_sample_submission.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        stage2IDs.append(row['id'])

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

filePathPrefix2 = '/home/zdestefa/data/segFilesStage2VGG/vgg19Feats_'
filePathPrefix1 = '/home/zdestefa/data/segFilesResizedVGG19to4096/resnetFeats_'

def getFeatureData(id,pathPrefix):
    fileName = pathPrefix + id + '.npy'
    featData = np.load(fileName)
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData, axis=0)  # this causes potential overfit. should remove
    return outVec


def train_xgboost():

    print('Train/Validation Data being obtained')
    #x = np.array([np.mean(getVolData(id), axis=0) for id in trainTestIDs.tolist()])
    numPatients = len(trainTestIDs)+len(validationIDs)
    x = np.zeros((numPatients,numConcatFeats))
    y = np.zeros((numPatients))
    ind = 0
    for id in trainTestIDs:
        x[ind,:] = getFeatureData(id,filePathPrefix1)
        y[ind] = float(trainTestLabels[ind])
        ind = ind+1
    ind2=0
    for id in validationIDs:
        x[ind,:] = getFeatureData(id,filePathPrefix1)
        y[ind] = float(validationLabels[ind2])
        ind2 = ind2+1
        ind = ind+1
    print('Finished getting train/test data')
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=20,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True,
            eval_metric='logloss', early_stopping_rounds=100)

    yHatValidation = clf.predict(val_x)
    [falsePos, truePos, posThresholds] = roc_curve(val_y, yHatValidation, pos_label=1)
    [falseNeg, trueNeg, negThresholds] = roc_curve(val_y, 1-yHatValidation, pos_label=0)
    posThreshAcc = np.zeros((len(posThresholds)))
    ind=0
    negThreshAcc = np.zeros((len(negThresholds)))
    for thresh in posThresholds:
        posThreshAcc[ind] = accuracy_score(val_y,(yHatValidation>thresh))
        ind = ind + 1
    ind=0
    for thresh in negThresholds:
        negThreshAcc[ind] = accuracy_score(val_y,((1-yHatValidation)>thresh))
        ind = ind + 1
    rocSaveFolder = '/home/zdestefa/rocCurveFiles'
    np.save(os.path.join(rocSaveFolder, 'falsePos.npy'), falsePos)
    np.save(os.path.join(rocSaveFolder, 'truePos.npy'), truePos)
    np.save(os.path.join(rocSaveFolder, 'posThresholds.npy'), posThresholds)
    np.save(os.path.join(rocSaveFolder, 'falseNeg.npy'), falseNeg)
    np.save(os.path.join(rocSaveFolder, 'trueNeg.npy'), trueNeg)
    np.save(os.path.join(rocSaveFolder, 'negThresholds.npy'), negThresholds)
    np.save(os.path.join(rocSaveFolder, 'posThreshAcc.npy'), posThreshAcc)
    np.save(os.path.join(rocSaveFolder, 'negThreshAcc.npy'), negThreshAcc)

    return clf


def make_submit():
    clf = train_xgboost()

    print('Kaggle Test Data being obtained')
    x2 = np.zeros((len(stage2IDs), numConcatFeats))
    ind = 0
    for id in stage2IDs:
        x2[ind, :] = getFeatureData(id,filePathPrefix2)
        ind = ind + 1
    print('Finished getting kaggle test data')

    pred = clf.predict(x2)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
    fileName = 'submissions/STAGE2_VGG19PlusXGBoost_' + st + '.csv'

    with open(fileName, 'w') as csvfile:
        fieldnames = ['id', 'cancer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ind in range(len(stage2IDs)):
            writer.writerow({'id': stage2IDs[ind], 'cancer': str(pred[ind])})


if __name__ == '__main__':
    #calc_featuresA()
    make_submit()