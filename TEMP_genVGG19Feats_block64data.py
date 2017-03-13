import numpy as np
import os

import scipy.io as sio

fileFolder = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet'

def getVolData(fileP):
    curMATcontent = sio.loadmat(os.path.join(fileFolder,fileP))
    volData = curMATcontent["huBlocks"]
    numHU = len(volData)
    cancerData = np.reshape(curMATcontent["outputCancerScores"],numHU)
    noduleData = np.reshape(curMATcontent["outputNoduleRadii"],numHU)
    lungData = np.reshape(curMATcontent["outputNumLungPixels"],numHU)
    return cancerData,noduleData,lungData

def genResNetFeatFile(fileP):
    patID = fileP[9:len(fileP) - 4]
    cancer,nod,lung = getVolData(fileP)
    print(cancer)
    print(nod)
    print(lung)

def calc_featuresA():
    filesToProcess = os.listdir(fileFolder)
    numFiles = len(filesToProcess)
    randomInds = np.random.choice(numFiles,20,replace=False)
    for curInd in randomInds:
        print('Obtaining features for file_' + str(curInd) + '_of_' + str(numFiles))
        genResNetFeatFile(filesToProcess[curInd])

if __name__ == '__main__':
    calc_featuresA()