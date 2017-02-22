import numpy as np
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

def getVolData(patID):
    patFile = "/home/zdestefa/data/segFilesResizedAll/resizedSegDCM_" + patID + ".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["resizedDCM"]
    return volData.astype('float32')

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)

net2 = Model(input=origNet.input,output=origNet.get_layer('flatten').output)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)



def genResNetFeatFile(id):
    fileName2 = 'data/segFilesResizedVGG19to25088/resnetFeats_' + id + '.npy'
    fileName3 = 'data/segFilesResizedVGG19to4096/resnetFeats_' + id + '.npy'
    curData = getVolData(id)
    curDataReshape = np.reshape(curData,(1,256,256,100))
    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, curData.shape[2] - 3, 3):
        tmp = []
        for j in range(3):
            img2 = curData[i + j]
            img = imresize(img2,(224,224))
            tmp.append(img)
        tmp = np.array(tmp)
        batch.append(np.array(tmp))
    batch = np.array(batch)
    feats2 = net2.predict(batch)
    feats3 = net3.predict(batch)

    np.save(fileName2, feats2)
    np.save(fileName3, feats3)

def calc_featuresA():
    total = len(trainTestIDs)+len(validationIDs)
    curInd = 0
    for id in trainTestIDs:
        curInd = curInd + 1
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(total))
        genResNetFeatFile(id)
    for id in validationIDs:
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(total))
        curInd = curInd + 1
        genResNetFeatFile(id)


if __name__ == '__main__':
    calc_featuresA()