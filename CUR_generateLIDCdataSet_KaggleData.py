import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom

curDir = '/home/zdestefa/data/rawHUdata'

print('Loading Binary Array')

matFiles = os.listdir(curDir)



for fInd in range(35,len(matFiles)):
    print('Now Processing File ' + str(fInd) + ' of ' + str(len(matFiles)))

    curFile = os.path.join(curDir,matFiles[fInd])
    patID = matFiles[fInd][7:len(matFiles[fInd])-4]

    #huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    #resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    #huDataFileNm = huDataFilePrefix+patID+'.npy'
    #resizeFileNm = resizeTupleFilePrefix+patID+'.npy'

    matData = sio.loadmat(curFile)
    rawHUdata = matData['dcmArrayHU']

    blockDim = 64
    blockDimHalf = 32
    maxX = rawHUdata.shape[0]-(blockDim+2)
    maxY = rawHUdata.shape[1]-(blockDim+2)
    maxZ = rawHUdata.shape[2]-(blockDim+2)
    rangeX = range(34,maxX,32)
    rangeY = range(34,maxY,32)
    rangeZ = range(34, maxZ, 32)
    numGridPts = len(rangeX)*len(rangeY)*len(rangeZ)
    xyzRange = np.meshgrid(rangeX,rangeY,rangeZ)

    xValues = np.reshape(xyzRange[0],numGridPts)
    yValues = np.reshape(xyzRange[1],numGridPts)
    zValues = np.reshape(xyzRange[2],numGridPts)

    print('Matrix Conversion done. Doing Sliding Window...')

    huBlocks = []
    numPossibleLungThreshold = np.floor(64 * 64 * 64 * 0.35)
    for ii in range(len(xValues)):
        curPtR = xValues[ii]
        curPtC = yValues[ii]
        curPtS = zValues[ii]

        xMin = curPtR-blockDimHalf
        xMax = xMin + blockDim
        yMin = curPtC-blockDimHalf
        yMax = yMin + blockDim
        zMin = curPtS-blockDimHalf
        zMax = zMin + blockDim
        currentHUdataBlock = rawHUdata[xMin:xMax,yMin:yMax,zMin:zMax]
        numPossibleLung = np.sum(np.logical_and(currentHUdataBlock > -1200, currentHUdataBlock < -700))
        if(numPossibleLung>numPossibleLungThreshold ):
            print("Num Lung: " + str(numPossibleLung))
            huBlocks.append(currentHUdataBlock)

    outFile = '/home/zdestefa/data/huBlockDataSetKaggleOrigSize/huBlocks_'+patID+'.mat'

    huBlocksOutput = np.zeros((len(huBlocks),64,64,64))
    curI = 0
    for block in huBlocks:
        huBlocksOutput[curI,:,:,:]=block
        curI = curI + 1
    sio.savemat(outFile, {"huBlocksOutput": huBlocksOutput})
