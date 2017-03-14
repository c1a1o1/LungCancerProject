import numpy as np
import os
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
import cv2
# import mxnet as mx
# import pandas as pd
# from sklearn import cross_validation
# import xgboost as xgb
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

from keras.applications.vgg19 import VGG19
import scipy.io as sio
from scipy.misc import imresize

fileFolder = '/home/zdestefa/data/huBlockDataSetKaggleOrigSize'

def getVolData(fileP):
    curMATcontent = sio.loadmat(os.path.join(fileFolder,fileP))
    volData = curMATcontent["huBlocksOutput"]
    return volData.astype('float32')

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

#net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)

def genResNetFeatFile(fileP):
    patID = fileP[9:len(fileP) - 4]
    huBlocks= getVolData(fileP)
    rPrefix = 'resnetFeats_'
    folder4096prefix = 'data/blockFilesResizedVGG19to4096Kaggle/'
    blockNum=0
    for nodInd in range(len(huBlocks)):
        currentBlock = huBlocks[nodInd,:,:,:]
        noduleSuffix = '_Block_' + str(blockNum)
        blockNum = blockNum+1
        fileName3 = folder4096prefix + rPrefix +patID + noduleSuffix
        print("obtaining resnet data for block: " + str(blockNum))
        feats3 = getResNetData(currentBlock)
        np.save(fileName3, feats3)


cnt = 0
dx = 40
ds = 512

def getResNetData(curData):
    curImg = curData
    #curImg[curImg==-2000]=0
    batch = []
    #for i in range(0, curData.shape[0] - 3, 3):
    for i in range(curData.shape[2]):
        tmp = []
        for j in range(3):
            img = curImg[i]
            #
            #NOTE: DO NOT DO THE EQUALIZEHIST PROCEDURE
            #   RESULTS ARE DRAMATICALLY BETTER WITHOUT IT
            #   WENT FROM 0.52 LOG LOSS TO 0.33 LOG LOSS
            #
            #RESULTS ARE ALSO MUCH BETTER REPLACIATING THE IMAGE
            #   IN EACH CHANNEL RATHER THAN TRY TO COMBINE
            #   IMAGES IN THE COLOR CHANNELS
            #
            #img = 255.0 / np.amax(img) * img
            #img = cv2.equalizeHist(img.astype(np.uint8))
            #img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    feats3 = net3.predict(batch)
    return feats3

def calc_featuresA():
    filesToProcess = os.listdir(fileFolder)
    numFiles = len(filesToProcess)
    for curInd in range(numFiles):
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(numFiles))
        genResNetFeatFile(filesToProcess[curInd])



if __name__ == '__main__':
    calc_featuresA()