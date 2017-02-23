import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb

trainTestIDs = []
trainTestLabels = []
validationIDs = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])

with open('stage1_sample_submission.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        validationIDs.append(row['id'])

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

def getFeatureData(id):
    fileName = '/home/zdestefa/data/segFilesResizedVGG19to4096/resnetFeats_' + id + '.npy'
    featData = np.load(fileName)
    outVec = np.zeros((1,numConcatFeats))
    outVec[0, 0:numGivenFeat] = np.mean(featData, axis=0)
    outVec[0, numGivenFeat:numGivenFeat * 2] = np.max(featData, axis=0)  # this causes potential overfit. should remove
    outVec[0, numGivenFeat * 2:numGivenFeat * 3] = np.min(featData, axis=0)  # this causes potential overfit. should remove
    return outVec


def train_xgboost():

    print('Train/Validation Data being obtained')
    #x = np.array([np.mean(getVolData(id), axis=0) for id in trainTestIDs.tolist()])

    x = np.zeros((len(trainTestIDs),numConcatFeats))
    ind = 0
    for id in trainTestIDs:
        x[ind,:] = getFeatureData(id)
        ind = ind+1
        #x.append(np.array([np.mean(featData,axis=0)]))
    print('Finished getting train/test data')
    #y = np_utils.to_categorical(trainTestLabels, 2)
    y = [float(lab) for lab in trainTestLabels]
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    # clf = xgb.XGBRegressor(max_depth=30,
    #                        n_estimators=1500,
    #                        min_child_weight=9,
    #                        learning_rate=0.05,
    #                        nthread=8,
    #                        subsample=0.80,
    #                        colsample_bytree=0.80,
    #                        seed=4242)

    #change the max depth
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


if __name__ == '__main__':
    #calc_featuresA()
    make_submit()