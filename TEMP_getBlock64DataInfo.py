import numpy as np
import csv
import time
import datetime

from sklearn import cross_validation
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import scipy.io as sio

numGivenFeat=4096
numConcatFeats = numGivenFeat*3

dataFolder = '/home/zdestefa/data/blockFilesResizedVGG19to4096'

def getBlockInfoData(resNetFileNm):
    blockDataSuffix = resNetFileNm[12:len(resNetFileNm)]
    blockDataFileNm = 'blockData_'+blockDataSuffix
    blockDataFile = os.path.join(dataFolder,blockDataFileNm)
    infoArray = np.load(blockDataFile)
    cancerScore = infoArray[0]
    noduleRadius = infoArray[1]
    lungPixelNum = infoArray[2]
    return cancerScore,noduleRadius,lungPixelNum

print('Train/Validation Data being obtained')
files = os.listdir(dataFolder)

resNetFiles = [file1 for file1 in files if file1.startswith("resnetFeats")]
cancerScores = np.zeros(len(resNetFiles))
nodRadii = np.zeros(len(resNetFiles))
lungNums = np.zeros(len(resNetFiles))
for fInd in range(len(resNetFiles)):
    print('Processing file ' + str(fInd) + ' of ' + str(len(resNetFiles)))
    cancerScores[fInd],nodRadii[fInd],lungNums[fInd] = getBlockInfoData(resNetFiles[fInd])

infoDataFile = '/home/zdestefa/data/TEMP_block64Data.mat'
sio.savemat(infoDataFile,{"cancerScores":cancerScores,"nodRadii":nodRadii,"lungNums":lungNums})

cancerPts = np.where(cancerScores>2)

print('Num Cancer Blocks: ' + str(len(cancerPts)))

cancerPtsUse = cancerPts

noCancerPts = np.where(cancerScores<1)

hist, bins = np.histogram(lungNums[noCancerPts], bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
"""
lungBlocks = np.where(lungNums>(64*64*64*0.05))
otherBlocksInclude = [ptNum for ptNum in noCancerPts if ptNum in lungBlocks]
otherPtsUse = otherBlocksInclude

numCancer = 500
numOther = 500
if(numCancer < len(cancerPts)):
    cancerInds = np.random.choice(range(len(cancerPts)), size=numCancer, replace=False)
    cancerPtsUse = cancerPts[cancerInds]
if(numOther < len(otherBlocksInclude)):
    otherInds = np.random.choice(range(len(otherBlocksInclude)),size=numOther,replace=False)
    otherPtsUse = otherBlocksInclude[otherInds]

numDataPts = len(cancerPtsUse)+len(otherPtsUse)
x = np.zeros((numDataPts,numConcatFeats))
y = np.zeros(numDataPts)
"""



