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

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols,img_sli)
    input_shape2 = (1, img_sli, 1000 )
else:
    input_shape = (img_rows, img_cols,img_sli, 1)
    input_shape2 = (img_sli, 1000, 1)

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

def myAlexNet(weights_path=None):
    inputs = Input(shape=(3, 227, 227))
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
    #dense_3 = Dropout(0.5)(dense_2)
    #dense_3 = Dense(1000, name='dense_3')(dense_3)
    # prediction = Activation("softmax", name="softmax")(dense_3)
    #model = Model(input=inputs, output=prediction)
    model = Model(input=inputs, output=dense_2)
    if weights_path:
        model.load_weights(weights_path)
    return model
#Here is code from a 3D CNN example on the following blog:
#   http://learnandshare645.blogspot.in/2016/06/3d-cnn-in-keras-action-recognition.html
#
#Good initial CNN tutorial:
#   http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

#ourAlexModel = convnet('alexnet',weights_path='alexnet_weights.h5')
ourAlexModel = myAlexNet()
originalAlexModel = convnet('alexnet', weights_path='alexnet_weights.h5', heatmap=False)
#alexmodel = convnet('alexnet', heatmap=False)
for layer, mylayer in zip(originalAlexModel.layers, ourAlexModel.layers):
  print(layer.name)
  #if mylayer.name == 'mil_1':
  if mylayer.name == 'flatten':
    break
  else:
    weightsval = layer.get_weights()
    print(len(weightsval))
    mylayer.set_weights(weightsval)

ourAlexModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#TODO CHANGE THIS TO DEPEND ON WHETHER ALEXNET HAS WEIGHTS

print("Now predicting training/test data from AlexNet layers")
trainTestDataAlex = ourAlexModel.predict_generator(dataGenerator2D(trainTestIDs),val_samples=len(trainTestIDs)*img_sli)

print("Now predicting validation data from AlexNet layers")
validationDataAlexNet = ourAlexModel.predict_generator(dataGenerator2D(validationIDs),val_samples=len(validationIDs)*img_sli)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'cnnPredictions/alexNetFeaturesFrom_'+st+'.mat'
sio.savemat(fileName,mdict={'trainTestDataAlex':trainTestDataAlex,'validationDataAlexNet':validationDataAlexNet})
"""
mat_contents = sio.loadmat('cnnPredictions/alexNetFeaturesCurrent.mat')
validationDataAlexNet = mat_contents['validationDataAlexNet']
trainTestDataAlex = mat_contents['trainTestDataAlex']
"""

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
print("Now resizing training and test data...")
numValidPts = len(validationIDs)
numTrainTestPts = len(trainTestIDs)
numCat = 1000
alexNetValid = np.zeros((numValidPts, img_sli,numCat))
alexNetTrainTest = np.zeros((numTrainTestPts, img_sli,numCat))
volInd=0
sliceInd=0
for ind in range(img_sli*numTrainTestPts):
    if(volInd<numValidPts):
        alexNetValid[volInd, sliceInd,:] = validationDataAlexNet[ind,:]
    alexNetTrainTest[volInd, sliceInd,:] = trainTestDataAlex[ind, :]

    sliceInd = sliceInd+1
    if(sliceInd>=img_sli):
        sliceInd=0
        volInd=volInd+1

alexTrain,alexTest,Ytrain,Ytest = train_test_split(alexNetTrainTest,Ydata,test_size=0.2,random_state=42)

Ytrain = np_utils.to_categorical(Ytrain, nb_classes)
Ytest = np_utils.to_categorical(Ytest, nb_classes)

if K.image_dim_ordering() == 'th':
    alexTrain = alexTrain.reshape(alexTrain.shape[0], 1, img_sli, numCat)
    alexTest = alexTest.reshape(alexTest.shape[0], 1, img_sli, numCat)
    alexNetValid=alexNetValid.reshape(alexNetValid.shape[0], 1, img_sli, numCat)
    input_shape = (1, img_sli, 1000)
else:
    alexTrain = alexTrain.reshape(alexTrain.shape[0], img_sli, 1000, 1)
    alexTest = alexTest.reshape(alexTest.shape[0], img_sli, 1000, 1)
    alexNetValid = alexNetValid.reshape(alexNetValid.shape[0], img_sli, numCat, 1)
    input_shape = (img_sli, 1000, 1)

print("Now constructing the new CNN")
postAlexModel = Sequential()

postAlexModel.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape2))
postAlexModel.add(Activation('relu'))
postAlexModel.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
postAlexModel.add(Activation('relu'))
postAlexModel.add(MaxPooling2D(pool_size=pool_size2))
postAlexModel.add(Dropout(0.25))

postAlexModel.add(Flatten())
postAlexModel.add(Dense(128))
postAlexModel.add(Activation('relu'))
postAlexModel.add(Dropout(0.5))
postAlexModel.add(Dense(nb_classes))
postAlexModel.add(Activation('softmax'))

postAlexModel.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

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