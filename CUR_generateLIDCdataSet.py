import numpy as np
import os
import scipy.io as sio
import numpy.matlib
from scipy.ndimage.interpolation import zoom

curDir = '/home/zdestefa/LUNA16/data/DOI_modNoduleInfo'

print('Loading Binary Array')

matFiles = os.listdir(curDir)

for fInd in range(len(matFiles)):
    print('Now Processing File ' + str(fInd) + ' of ' + str(len(matFiles)))

    curFile = os.path.join(curDir,matFiles[fInd])
    patID = matFiles[fInd][18:len(matFiles[fInd])-4]

    huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    huDataFileNm = huDataFilePrefix+patID+'.npy'
    resizeFileNm = resizeTupleFilePrefix+patID+'.npy'


    initPointsForSample = []
    cancerLikelihoods = []
    noduleRadii = []

    matData = sio.loadmat(curFile)
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

    huData = np.load(huDataFileNm)
    resizeData = np.load(resizeFileNm)

    print('Binary Array Loaded. Converting to Matrix...')

    huDataResized = zoom(huData,resizeData)

    #samples random points to use as center of 64x64x64 blocks
    #make sure they are at least 24 pixels apart
    #   this allows some overlap but not too much overlap

    numSampledPts = 4500
    blockDim = 64
    blockDimHalf = 32
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
        #this is where I convert from center coordinates specified
        #   to row, column, slice coordinates
        #Upper Left is (0,0) in x-y plane
        #   Thus (NumRows-y) is row coordinate
        #   x will be col coordinate
        rowCoord = np.floor(huDataResized.shape[1]-(initPointsForSample[ii][1]*resizeData[1]))
        colCoord = np.floor(initPointsForSample[ii][0]*resizeData[0])
        sliCoord = np.floor(initPointsForSample[ii][2]*resizeData[2])

        #ensures that row, column, slice are within inner block
        rowCoord = max(rowCoord,blockDimHalf+2)
        rowCoord = min(rowCoord,huDataResized.shape[0]-(blockDimHalf+2))

        colCoord = max(colCoord, blockDimHalf + 2)
        colCoord = min(colCoord, huDataResized.shape[1] - (blockDimHalf + 2))

        sliCoord = max(sliCoord, blockDimHalf + 2)
        sliCoord = min(sliCoord, huDataResized.shape[2] - (blockDimHalf + 2))

        samplePointsUse[ii,:] = [rowCoord,colCoord,sliCoord]
    for ii in range(numSampledPts):
        samplePointsUse[ii+numInitPts,:] = finalSamples[ii,:]

    print('Sampling Finished. Now checking validity of each sample')

    prismValid = np.ones(huDataResized.shape)
    finalPts = []
    huBlocks = []
    #outputNumLungPixels = []
    outputCancerScores = []
    outputNoduleRadii = []
    numPossibleLungThreshold = np.floor(64 * 64 * 64 * 0.35)
    numSamplesMax = 30 #max number of sampled blocks
    numSamplesCur = 0
    for ii in range(samplePointsUse.shape[0]):
        curPt = samplePointsUse[ii,:]
        curPtR = int(curPt[0])
        curPtC = int(curPt[1])
        curPtS = int(curPt[2])
        cancerNum = 0
        nodRadius = 0
        if(ii<len(initPointsForSample)):
            cancerNum = cancerLikelihoods[ii]
            nodRadius = noduleRadii[ii]
        if(prismValid[curPtR,curPtC,curPtS]>0):
            finalPts.append(curPt)
            xMin = curPtR-blockDimHalf
            xMax = xMin + blockDim
            yMin = curPtC-blockDimHalf
            yMax = yMin + blockDim
            zMin = curPtS-blockDimHalf
            zMax = zMin + blockDim
            currentHUdataBlock = huDataResized[xMin:xMax,yMin:yMax,zMin:zMax]
            #print("yMin: " + str(yMin) + "; yMax:" + str(yMax))
            #print("huDataResizedDims:" + str(huDataResized.shape))
            print("HuBlockDims: " + str(currentHUdataBlock.shape))
            numPossibleLung = np.sum(np.logical_and(currentHUdataBlock > -1200, currentHUdataBlock < -700))
            if(numPossibleLung>numPossibleLungThreshold or ii<len(initPointsForSample)):
                prismValid[xMin:xMax, yMin:yMax, zMin:zMax] = 0
                print("Cancer: " + str(cancerNum) + "Radius: " + str(nodRadius) + "Num Lung: " + str(numPossibleLung))
                huBlocks.append(currentHUdataBlock)
                outputCancerScores.append(cancerNum)
                outputNoduleRadii.append(nodRadius)
                #outputNumLungPixels.append(numPossibleLung)
                #print("outputNumLungPixels: " + str(outputNumLungPixels))
                numSamplesCur = numSamplesCur + 1
                if(numSamplesCur>numSamplesMax):
                    break

    outputC = np.array(outputCancerScores)
    numCancer = np.sum(outputC>0)
    numNoCancer = len(outputC)-numCancer
    print("Final Report, Num Cancer:" + str(numCancer) + "  Num No Cancer:" + str(numNoCancer) )

    outFile = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet/huBlocks_'+patID+'.mat'
    # sio.savemat(outFile,{
    #     "huBlocks":huBlocks,
    #     "outputCancerScores":outputCancerScores,
    #     "outputNoduleRadii":outputNoduleRadii,
    #     "outputNumLungPixels":outputNumLungPixels})
    huBlocksOutput = np.zeros((len(huBlocks),64,64,64))
    curI = 0
    for block in huBlocks:
        huBlocksOutput[curI,:,:,:]=block
        curI = curI + 1
    print("HU Blocks Shape" + str(huBlocksOutput.shape))
    sio.savemat(outFile, {
        "huBlocksOutput": huBlocksOutput,
        "outputCancerScores": outputCancerScores,
        "outputNoduleRadii": outputNoduleRadii})
