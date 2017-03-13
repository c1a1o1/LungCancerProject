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

fileFolder = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet'

def getVolData(fileP):
    curMATcontent = sio.loadmat(os.path.join(fileFolder,fileP))
    volData = curMATcontent["huBlocks"]
    numHU = len(volData)
    cancerData = np.reshape(curMATcontent["outputCancerScores"],numHU)
    noduleData = np.reshape(curMATcontent["outputNoduleRadii"],numHU)
    lungData = np.reshape(curMATcontent["outputNumLungPixels"],numHU)
    return volData.astype('float32'),cancerData,noduleData,lungData

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

#net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)

def genResNetFeatFile(fileP):
    patID = fileP[9:len(fileP) - 4]
    huBlocks,cancer,nod,lung = getVolData(fileP)
    folder25088prefix = 'data/blockFilesResizedVGG19to25088/'
    rPrefix = 'resnetFeats_'
    bPrefix = 'blockData_'
    folder4096prefix = 'data/blockFilesResizedVGG19to4096/'
    for nodInd in range(len(huBlocks)):
        curData = huBlocks[nodInd,:,:,:]
        cancerNum = cancer[nodInd]


        noduleSuffix = '_Block_' + str(nodInd)
        if(cancerNum>0):
            labelSuffix = '_label0.npy'
        else:
            labelSuffix = '_label1.npy'

        #fileName2 = folder25088prefix + rPrefix + patID + noduleSuffix + labelSuffix
        fileName3 = folder4096prefix + rPrefix +patID + noduleSuffix + labelSuffix
        # fileName2A = folder25088prefix + bPrefix + patID + noduleSuffix + labelSuffix
        #fileName3A = folder4096prefix + bPrefix + patID + noduleSuffix + labelSuffix

        feats3 = getResNetData(curData)
        #np.save(fileName2, feats2)
        np.save(fileName3, feats3)

        #outputBlockData = [cancer[nodInd], nod[nodInd], lung[nodInd]]
        #np.save(fileName2A, outputBlockData)
        #np.save(fileName3A, outputBlockData)

cnt = 0
dx = 40
ds = 512

def getResNetData(curData):
    curImg = curData
    curImg[curImg==-2000]=0
    batch = []
    for i in range(curData.shape[2]):
        tmp = []
        img = curImg[i]
        img = 255.0 / np.amax(img) * img
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = img[dx: ds - dx, dx: ds - dx]
        img = cv2.resize(img, (224, 224))
        for j in range(3):
            tmp.append(img)
        tmp = np.array(tmp)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    #feats2 = net2.predict(batch)
    feats3 = net3.predict(batch)
    return feats3

def calc_featuresA():
    filesToProcess = os.listdir(fileFolder)
    numFiles = len(filesToProcess)
    # for fileP in filesToProcess:
    for curInd in range(800,numFiles):
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(numFiles))
        genResNetFeatFile(filesToProcess[curInd])



if __name__ == '__main__':
    calc_featuresA()