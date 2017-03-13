import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom

curDir = '/home/zdestefa/LUNA16/data/DOI_modNoduleInfo'

print('Loading Binary Array')

matFiles = os.listdir(curDir)
sInds = np.random.choice(len(matFiles),40,replace=False)

for fInd in range(1):
    fileA = 'noduleInformation_1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mat'
    curFile = os.path.join(curDir,fileA)
    patID = matFiles[fInd][18:len(fileA)-4]
    huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    huDataFileNm = huDataFilePrefix+patID+'.npy'
    resizeFileNm = resizeTupleFilePrefix+patID+'.npy'


    initPointsForSample = []
    cancerLikelihoods = []
    noduleRadii = []

    matData = sio.loadmat(curFile)
    print(matData['nodRadii'])

    rawRadArray = matData['nodRadii']
    radArray = np.reshape(rawRadArray,len(rawRadArray))
    rawMalArray = matData['malignancies']
    malArray = np.reshape(rawMalArray,len(rawMalArray))
    regCenters = matData['nodCenters']

    #obtain a list sorted by malignancy number first, then nodule radius
    dtype = [('mal', int), ('rad', float)]
    values = [(malArray[jj], radArray[jj]) for jj in range(len(radArray))]
    malRadArray = np.array(values, dtype=dtype)
    indsUse = list(reversed(np.argsort(malRadArray,order=['mal','rad'])))

    for nInd in indsUse:
        initPointsForSample.append(regCenters[nInd])
        cancerLikelihoods.append(malArray[nInd])
        noduleRadii.append(radArray[nInd])

    print('Cancer and Radius Info:')
    print(cancerLikelihoods)
    print(noduleRadii)

    huData = np.load(huDataFileNm)
    resizeData = np.load(resizeFileNm)

    print('Binary Array Loaded. Converting to Matrix...')

    huDataResized = zoom(huData,resizeData)

    #samples random points to use as center of 64x64x64 blocks
    #make sure they are at least 24 pixels apart
    #   this allows some overlap but not too much overlap

    numSampledPts = 4500
    blockDim = 64
    numInXrange = huDataResized.shape[0]-(blockDim+2)
    numInYrange = huDataResized.shape[1]-(blockDim+2)
    numInZrange = huDataResized.shape[2]-(blockDim+2)
    matrixToMultBy = np.matlib.repmat([numInXrange,numInYrange,numInZrange],numSampledPts,1)

    print('Matrix Conversion done. Doing Sampling...')

    initSamples = np.random.rand(numSampledPts,3)
    multipliedSamples = np.multiply(matrixToMultBy,initSamples)
    finalSamples = np.floor(np.add(multipliedSamples,blockDim/2 + 2))

    numInitPts = len(initPointsForSample)
    samplePointsUse = np.zeros((numSampledPts+numInitPts,3))
    for ii in range(numInitPts):
        samplePointsUse[ii,:] = [
            initPointsForSample[ii][0]*resizeData[0],
            initPointsForSample[ii][1]*resizeData[1],
            initPointsForSample[ii][2]*resizeData[2]
        ]
    for ii in range(numSampledPts):
        samplePointsUse[ii+numInitPts,:] = finalSamples[ii,:]

    print('Sampling Finished. Now checking validity of each sample')

    prismValid = np.ones(huDataResized.shape)
    finalPts = []
    huDataBlocks = []
    outputNumLungPixels = []
    huBlocks = []
    outputCancerScores = []
    outputNoduleRadii = []
    numPossibleLungThreshold = np.floor(64*64*64*0.01)
    for ii in range(samplePointsUse.shape[0]):
        curPt = samplePointsUse[ii,:].astype('int32')
        curPtR = curPt[0]
        curPtC = curPt[1]
        curPtS = curPt[2]
        cancerNum = 0
        nodRadius = 0

        if(ii<len(initPointsForSample)):
            cancerNum = cancerLikelihoods[ii]
            nodRadius = noduleRadii[ii]
            print("Cancer:" + str(cancerNum))
            print("Rad: " + str(nodRadius))
            print("Valid Prism: " + str(prismValid[curPtR,curPtC,curPtS]))
        if(prismValid[curPtR,curPtC,curPtS]>0):
            if(cancerNum>0 or nodRadius>0):
                print("Cancer: " + str(cancerNum))
                print("Radius: " + str(nodRadius))
            finalPts.append(curPt)
            xMin = int(curPtR-blockDim/2)
            xMax = xMin +  blockDim
            yMin = int(curPtC-blockDim/2)
            yMax = yMin + blockDim
            zMin = int(curPtS-blockDim/2)
            zMax = zMin + blockDim
            prismValid[xMin:xMax,yMin:yMax,zMin:zMax] = 0
            outputCancerScores.append(cancerNum)
            outputNoduleRadii.append(nodRadius)

    print(outputCancerScores)
    print(outputNoduleRadii)


