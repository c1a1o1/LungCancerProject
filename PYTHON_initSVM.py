'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
import csv
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn import svm

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#print(X_train.shape)

matFiles = []
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


#TODO: CHANGE WHEN DONE TESTING
numTrainTest = 50

numFeats = 256*256*100
Xdata = np.zeros((numTrainTest,numFeats))
Ydata = np.zeros(numTrainTest)

numValid = len(validationIDs)
Xvalid = np.zeros((numValid,numFeats))

for ind in range(numTrainTest):
    patID = trainTestIDs[ind]
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_"+patID+".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    volDataVector = np.reshape(volData,(numFeats))
    Xdata[ind, :] = volDataVector
    Ydata[ind] = int(trainTestLabels[ind])
    print("Ind:" + str(ind))

for ind in range(numValid):
    patID = validationIDs[ind]
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_"+patID+".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    volDataVector = np.reshape(volData,(numFeats))
    Xvalid[ind, :] = volDataVector
    print("Ind2:" + str(ind))

Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xdata,Ydata,test_size=0.2,random_state=42)

clf = svm.SVC()
clf = clf.fit(Xtrain,Ytrain)

yHatTrain = clf.predict(Xtrain)
yHatTest = clf.predict(Xtest)
Yvalid = clf.predict(Xvalid)

trainError = sum(abs(yHatTrain-Ytrain))/len(Ytrain)
testError = sum(abs(yHatTest-Ytest))/len(Ytest)

print(trainError)
print(testError)

with open('submissionSVM.csv', 'w') as csvfile:
    fieldnames = ['id', 'cancer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(numValid):
        writer.writerow({'id': validationIDs[ind], 'cancer': str(Yvalid[ind])})