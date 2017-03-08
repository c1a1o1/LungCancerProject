import numpy as np
import os
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
#import cv2
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
    volData = curMATcontent["huBlocksNodule"]
    volData2 = curMATcontent["huBlocksNoNoduleLung"]
    return volData.astype('float32'),volData2.astype('float32')

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)

def genResNetFeatFile(fileP):
    patID = fileP[9:len(fileP) - 4]
    huBlocksNodule,huBlocksNoNodule = getVolData(fileP)
    for nodInd in range(huBlocksNodule.shape[0]):
        curData = huBlocksNodule[nodInd,:,:,:]
        fileName2 = 'data/blockFilesResizedVGG19to25088/resnetFeats_' + patID + '_Nodule_'+str(nodInd)+'.npy'
        fileName3 = 'data/blockFilesResizedVGG19to4096/resnetFeats_' + patID + '_Nodule_'+str(nodInd)+ '.npy'
        feats2,feats3 = getResNetData(curData)
        np.save(fileName2, feats2)
        np.save(fileName3, feats3)
    for nodInd in range(huBlocksNoNodule.shape[0]):
        curData = huBlocksNoNodule[nodInd,:,:,:]
        fileName2 = 'data/blockFilesResizedVGG19to25088/resnetFeats_' + patID + '_NoNodule_'+str(nodInd)+'.npy'
        fileName3 = 'data/blockFilesResizedVGG19to4096/resnetFeats_' + patID + '_NoNodule_'+str(nodInd)+ '.npy'
        feats2,feats3 = getResNetData(curData)
        np.save(fileName2, feats2)
        np.save(fileName3, feats3)

def getResNetData(curData):
    curDataReshape = np.reshape(curData, (1, 64, 64, 64))
    batch = []
    for i in range(0, curData.shape[2] - 3, 3):
        tmp = []
        for j in range(3):
            img2 = curData[i + j]
            img = imresize(img2, (224, 224))
            tmp.append(img)
        tmp = np.array(tmp)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    feats2 = net2.predict(batch)
    feats3 = net3.predict(batch)
    return feats2,feats3

def calc_featuresA():
    filesToProcess = os.listdir(fileFolder)
    curInd = 0
    for fileP in filesToProcess:
        curInd = curInd + 1
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(len(filesToProcess)))
        genResNetFeatFile(fileP)


if __name__ == '__main__':
    calc_featuresA()