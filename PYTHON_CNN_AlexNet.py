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

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols,img_sli)
else:
    input_shape = (img_rows, img_cols,img_sli, 1)

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

def lungProjectAlexNet():
    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation="relu", name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation="relu", name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation="relu", name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)

    prediction = dense_3
    #prediction = Activation("softmax", name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    return model

#Here is code from a 3D CNN example on the following blog:
#   http://learnandshare645.blogspot.in/2016/06/3d-cnn-in-keras-action-recognition.html
#
#Good initial CNN tutorial:
#   http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/


#alexmodel = convnet('alexnet')
alexmodel = lungProjectAlexNet()
alexmodel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

validationDataAlexNet = alexmodel.predict_generator(dataGenerator2D(validationIDs),val_samples=len(validationIDs)*img_sli)
trainTestDataAlex = alexmodel.predict_generator(dataGenerator2D(trainTestIDs),val_samples=len(trainTestIDs)*img_sli)


#	YVALIDPREDALEX WILL OUTPUT 19800X1000 ARRAY
#TODO: DO THE FOLLOWING
#	CONSTRUCT DATA SET OF 198 ARRAYS OF SIZE 100X1000 FOR SLICESxCATEGORIES
#	RUN A RANDOM FOREST CLASSIFIER 

# numValidPts = len(validationIDs)
# numTrainTestPts = len(trainTestIDs)
# alexNetValidationCats = np.zeros((numValidPts,img_sli))
# alexNetTrainTestCats = np.zeros((numTrainTestPts,img_sli))
# volInd=0
# sliceInd=0
# for ind in range(img_sli*numTrainTestPts):
#     if(volInd<numValidPts):
#         currentCategory = np.argmax(validationDataAlexNet[ind,:])
#         alexNetValidationCats[volInd,sliceInd] = currentCategory
#
#     currentCategory2 = np.argmax(trainTestDataAlex[ind, :])
#     alexNetTrainTestCats[volInd, sliceInd] = currentCategory2
#
#     sliceInd = sliceInd+1
#     if(sliceInd>=img_sli):
#         sliceInd=0
#         volInd=volInd+1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'cnnPredictions/cnnPredictionAlexNetFrom_'+st+'.mat'


sio.savemat(fileName,mdict={'trainTestDataAlex':trainTestDataAlex,'validationDataAlexNet':validationDataAlexNet})
#sio.savemat(fileName,mdict={'alexNetTrainTestCats':alexNetTrainTestCats,'alexNetValidationCats':alexNetValidationCats})
#sio.savemat(fileName,mdict={'alexNetValidationCats':alexNetValidationCats})


"""
sio.savemat('RES_randomProj.mat',mdict={'yHatTrainP':yHatTrainP,'yHatTestP':yHatTestP,'YvalidP':YvalidP,'Ytrain':Ytrain,'Ytest':Ytest})

Yvalid = YvalidP[:,1]

with open('submissionRandProjRandomForest.csv', 'w') as csvfile:
    fieldnames = ['id', 'cancer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(numValid):
        writer.writerow({'id': validationIDs[ind], 'cancer': str(Yvalid[ind])})

"""