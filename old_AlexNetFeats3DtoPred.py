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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
from scipy.misc import imread, imresize, imsave
import time
import datetime
import csv
from convnetskeras.convnets import preprocess_image_batch, convnet
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids


batch_size = 20
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows = 256
img_cols = 256
img_sli = 100

# number of convolutional filters to use
nb_filters = 10
# size of pooling area for max pooling
pool_size = (5,5,5)
pool_size2 = (5,5)
# convolution kernel size
kernel_size = (4,4,4)

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
trainingRatio = 0.90
numTrainTestAll = len(trainTestIDs)
#numTrainTestAll = 100

numTrain = int(np.floor(trainingRatio*numTrainTestAll))
numTest = numTrainTestAll-numTrain
numValid = len(validationIDs)

randInds = np.random.permutation(numTrainTestAll)
indsTrain = randInds[0:numTrain]
indsTest = randInds[numTrain:numTrainTestAll]

Ydata = np.zeros(numTrainTestAll)
for ind in range(numTrainTestAll):
    Ydata[ind] = int(trainTestLabels[ind])



def getVolData(patID):
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_" + patID + ".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    return volData.astype('float32')

def dataGenerator(patIDnumbers, patLabels, indsUse):
    while 1:
        for ind in range(len(indsUse)):
            patID = patIDnumbers[indsUse[ind]]
            XCur = getVolData(patID)
            if K.image_dim_ordering() == 'th':
                XCur = XCur.reshape(1, 1, img_rows, img_cols, img_sli)
            else:
                XCur = XCur.reshape(1, img_rows, img_cols, img_sli, 1)
            YCur = int(patLabels[indsUse[ind]])
            YUse = np_utils.to_categorical(YCur, nb_classes)
            #print("Ind:" + str(ind))
            yield (XCur.astype('float32'),YUse)

def validDataGenerator():
    while 1:
        for ind in range(len(validationIDs)):
            patID = validationIDs[ind]
            XCur = getVolData(patID)
            if K.image_dim_ordering() == 'th':
                XCur = XCur.reshape(1, 1, img_rows, img_cols, img_sli)
            else:
                XCur = XCur.reshape(1, img_rows, img_cols, img_sli, 1)
            #print("ValidInd:" + str(ind))
            yield (XCur.astype('float32'))

def dataGenerator2D(arrayIDs):
    while 1:
        for ind in range(len(arrayIDs)):
            print("Currently at slice ind: " + str(ind) + " of " + str(len(arrayIDs)))
            patID = arrayIDs[ind]
            XCur = getVolData(patID)
            for slice in range(img_sli):
                currentXslice = XCur[:,:,slice]
                currentInput = np.zeros((3,227,227))
                currentInput[0,:,:] = imresize(currentXslice,(227,227),'nearest')
                currentInput[1, :, :] = currentInput[0,:,:]
                currentInput[2, :, :] = currentInput[0, :, :]
                currentInput = currentInput.reshape(1,3,227,227)
                # print("ValidInd:" + str(ind))
                yield (currentInput.astype('float32'))


print("Currently loading validation data")
alexNetValid = np.load('transferData/alexNet3DValidation_current.npy')
print('Validation Array Loaded')
print(alexNetValid.shape)
print('Currently loading train test data...')
alexNetTrainTest = np.load('transferData/alexNet3DTrainTest_current.npy')
print('Training and Test Data loaded')
print(alexNetTrainTest.shape)

numRow = alexNetValid.shape[2]
numCol = alexNetValid.shape[3]
alexTrain,alexTest,Ytrain,Ytest = train_test_split(alexNetTrainTest,Ydata,test_size=0.2,random_state=42)

Ytrain = np_utils.to_categorical(Ytrain, nb_classes)
Ytest = np_utils.to_categorical(Ytest, nb_classes)

if K.image_dim_ordering() == 'th':
    alexTrain = alexTrain.reshape(alexTrain.shape[0], 1, img_sli, numRow,numCol)
    alexTest = alexTest.reshape(alexTest.shape[0], 1, img_sli, numRow,numCol)
    alexNetValid=alexNetValid.reshape(alexNetValid.shape[0], 1, img_sli, numRow,numCol)
    input_shape = (1, img_sli, numRow,numCol)
else:
    alexTrain = alexTrain.reshape(alexTrain.shape[0], img_sli, numRow,numCol, 1)
    alexTest = alexTest.reshape(alexTest.shape[0], img_sli, numRow,numCol, 1)
    alexNetValid = alexNetValid.reshape(alexNetValid.shape[0], img_sli, numRow,numCol, 1)
    input_shape = (img_sli, numRow,numCol, 1)

print("Now constructing the new CNN")
postAlexModel = Sequential()

postAlexModel.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1],kernel_size[2],
                        border_mode='valid',
                        input_shape=input_shape))
postAlexModel.add(MaxPooling3D(pool_size=pool_size))
postAlexModel.add(Dropout(0.2))
postAlexModel.add(Flatten())
#model.add(Dense(128, init='normal',activation='relu'))
postAlexModel.add(Dense(16, init='normal',activation='sigmoid'))
postAlexModel.add(Dense(nb_classes, init='normal',activation='softmax'))
postAlexModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

postAlexModel.fit(alexTrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(alexTest, Ytest))
score = postAlexModel.evaluate(alexTest, Ytest, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
prediction = postAlexModel.predict(alexNetValid)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'cnnPredictions/cnnPredictionAlexNetFrom_'+st+'.mat'
sio.savemat(fileName,mdict={'score':score,'prediction':prediction})
