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
from scipy.ndimage.interpolation import zoom

from keras.applications.resnet50 import ResNet50
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
    patFile = "/home/zdestefa/data/segFilesStage2/segDCM_" + patID + ".mat"
    curMATcontent = sio.loadmat(patFile)
    volData = curMATcontent["outputDCM"]
    zoomX = 256.0/volData.shape[0]
    zoomY = 256.0/volData.shape[1]
    zoomZ = 100.0/volData.shape[2]
    volDataOutput = zoom(volData,(zoomX,zoomY,zoomZ))
    return volDataOutput.astype('float32')

origNet = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
net3 = Model(input=origNet.input,output=origNet.get_layer('fc2').output)

def genResNetFeatFile(id):
    #fileName2 = 'data/segFilesResizedResNetAct48/resnetFeats_' + id + '.npy'
    fileName3 = 'data/segFilesStage2VGG/vgg19Feats_' + id + '.npy'
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
    feats3 = net3.predict(batch)
    np.save(fileName3, feats3)

def calc_featuresA():
    fileNames = os.listdir('/home/zdestefa/data/segFilesStage2')
    displayInd = 1
    for fileN in fileNames:
        print("Now Processing File " + str(displayInd) + " of " + str(len(fileNames)))
        if(fileN.endswith(".mat")):
            patientIDnum = fileN[7:len(fileN)-4]
            genResNetFeatFile(patientIDnum)
        displayInd = displayInd+1



if __name__ == '__main__':
    calc_featuresA()