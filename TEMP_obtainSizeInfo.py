import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom
import numpy as np
import os
#import dicom
#import glob
from matplotlib import pyplot as plt
import os
import csv
import cv2
import datetime
import numpy as np
from scipy.ndimage.interpolation import zoom
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import csv

from keras.applications.vgg19 import VGG19
import scipy.io as sio
from scipy.misc import imresize
numGivenFeat=4096
numConcatFeats = numGivenFeat*3


dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096'
dataFolder2 = '/home/zdestefa/data/blockFilesResizedVGG19to4096Kaggle'


curDir = '/home/zdestefa/data/rawHUdata'
curDir2 = '/home/zdestefa/data/volResizeInfo'

print('Loading Binary Array')

trainTestIDs = []
trainTestLabels = []
with open('stage1_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trainTestIDs.append(row['id'])
        trainTestLabels.append(row['cancer'])


matFiles = os.listdir(curDir)
matFiles2 = os.listdir(curDir2)

for fInd in range(len(matFiles)):
    print('Now Processing File ' + str(fInd) + ' of ' + str(len(matFiles)))

    curFile = os.path.join(curDir,matFiles[fInd])
    patID = matFiles[fInd][7:len(matFiles[fInd])-4]

    curFile2 = os.path.join(curDir2, matFiles2[fInd])
    #huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    #resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    #huDataFileNm = huDataFilePrefix+patID+'.npy'
    #resizeFileNm = resizeTupleFilePrefix+patID+'.npy'

    matData = sio.loadmat(curFile)
    matData2 = sio.loadmat(curFile2)
    rawHUdata = matData['dcmArrayHU']
    resizeData = np.reshape(matData2['resizeTuple'],3)

    huResize = zoom(rawHUdata,resizeData)
    prismValid = np.ones(huResize.shape)

    print("resizeData:" + str(resizeData))
    print('Orig Shape:' + str(rawHUdata.shape))
    print('Resized Shape: ' + str(huResize.shape))
    print('Prism Valid Shape: ' + str(prismValid.shape))

    if(huResize.shape[0] < 64 or huResize.shape[1] < 64 or huResize.shape[2] < 64):
        print('BAD SHAPE!!!')
        print('Bad File:' + matFiles[fInd])
        break
