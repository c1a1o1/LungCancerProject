import numpy as np
import os
import scipy.io as sio
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import zoom

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

def getRandomIndices(currentArrayLength,otherArrayLength,minNumSamples):
    numSamplesUse=otherArrayLength
    if(currentArrayLength<minNumSamples or currentArrayLength<=otherArrayLength):
        #if the current array has few entries, don't need to sample from it
        return range(currentArrayLength)
    elif(otherArrayLength<minNumSamples):
        #if the other array is small, then sample more than its length
        numSamplesUse = minNumSamples
    return np.random.choice(range(currentArrayLength), numSamplesUse, replace=False)

curDir = '/home/zdestefa/LUNA16/data/DOI_modNodule'
#curDir = 'D:\dev\git\LungCancerProject\DOI_modNodule'

print('Loading Binary Array')

matFiles = os.listdir(curDir)
minNumSamplesUse = 30

for fInd in range(len(matFiles)):
    print('Now Processing File ' + str(fInd) + ' of ' + str(len(matFiles)))

    curFile = os.path.join(curDir,matFiles[fInd])
    patID = matFiles[fInd][19:len(matFiles[fInd])-4]

    huDataFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/HUarray_'
    resizeTupleFilePrefix = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo/resizeTuple_'

    huDataFileNm = huDataFilePrefix+patID+'.npy'
    resizeFileNm = resizeTupleFilePrefix+patID+'.npy'

    matData = sio.loadmat(curFile)
    binArray = matData['finalOutputSparse'][0]
    initPointsForSample = []
    cancerLikelihoods = []

    for regionInd in range(len(matData['roiInfo'][0])):
        if(hasattr(matData['roiInfo'][0][0],'radius')):
            regRadius = matData['roiInfo'][0][regionInd]['radius'][0, 0][0][0]
            regCenter = [matData['roiInfo'][0][regionInd]['center'][0,0][0][0],
                         matData['roiInfo'][0][regionInd]['center'][0, 0][0][1],
                         matData['roiInfo'][0][regionInd]['center'][0, 0][0][2]
                         ]
            regCancerLikelihood = matData["roiInfo"][0][regionInd]["cancerLikelihood"][0,0][0][0]
            if(regCancerLikelihood>2 or regRadius>4):
                initPointsForSample.append(regCenter)
            cancerLikelihoods.append(regCancerLikelihood)


    huData = np.load(huDataFileNm)
    resizeData = np.load(resizeFileNm)

    print('Binary Array Loaded. Converting to Matrix...')

    binArrayDense = np.zeros((512,512,len(binArray)))
    for ii in range(len(binArray)):
        binArrayDense[:,:,ii]=binArray[ii].todense()

    huDataResized = zoom(huData,resizeData)
    binArrayResized = zoom(binArrayDense,resizeData)

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
            initPointsForSample[ii][2]*resizeData[2],
        ]
    for ii in range(numSampledPts):
        samplePointsUse[ii+numInitPts,:] = finalSamples[ii,:]

    print('Sampling Finished. Now checking validity of each sample')

    prismValid = np.ones(huDataResized.shape)
    finalPts = []
    huDataBlocks = []
    binaryDataBlocks = []
    binarySumValues = []
    numPossibleLungValues = []
    huBlocksNodule = []
    huBlocksNoNoduleLung = []
    huBlocksNoNoduleNoLung = []
    binBlocksNodule = []
    binBlocksNoNoduleLung = []
    binBlocksNoNoduleNoLung = []
    binarySumThreshold =100
    numPossibleLungThreshold = np.floor(64*64*64*0.01)
    for ii in range(samplePointsUse.shape[0]):
        curPt = samplePointsUse[ii,:]
        cancerNum = 0
        if(ii<len(initPointsForSample)):
            cancerNum = cancerLikelihoods[ii]
        curPtR = int(curPt[0])
        curPtC = int(curPt[1])
        curPtS = int(curPt[2])
        if(prismValid[curPtR,curPtC,curPtS]>0):
            finalPts.append(curPt)
            xMin = int(curPtR-blockDim/2)
            xMax = xMin +  blockDim
            yMin = int(curPtC-blockDim/2)
            yMax = yMin + blockDim
            zMin = int(curPtS-blockDim/2)
            zMax = zMin + blockDim
            prismValid[xMin:xMax,yMin:yMax,zMin:zMax] = 0
            currentHUdataBlock = huDataResized[xMin:xMax,yMin:yMax,zMin:zMax]
            huDataBlocks.append(currentHUdataBlock)
            currentBinaryDataBlock = binArrayResized[xMin:xMax,yMin:yMax,zMin:zMax]
            binarySum = np.sum(currentBinaryDataBlock)
            binaryDataBlocks.append(currentBinaryDataBlock)
            numPossibleLung = 0
            for block in currentHUdataBlock:
                for row in block:
                    for val in row:
                        if(val>-1200 and val<-700):
                            numPossibleLung = numPossibleLung+1
            print("Binary Sum: " + str(binarySum) + " NumLungPixels: " + str(numPossibleLung))
            if(binarySum>binarySumThreshold or cancerNum>2):
                huBlocksNodule.append(currentHUdataBlock)
                binBlocksNodule.append(currentBinaryDataBlock)
            elif(numPossibleLung>numPossibleLungThreshold):
                huBlocksNoNoduleLung.append(currentHUdataBlock)
                binBlocksNoNoduleLung.append(currentBinaryDataBlock)
            else:
                huBlocksNoNoduleNoLung.append(currentHUdataBlock)
                binBlocksNoNoduleNoLung.append(currentBinaryDataBlock)
            binarySumValues.append(binarySum)
            numPossibleLungValues.append(numPossibleLung)

    numNoduleBlocks = len(huBlocksNodule)
    numNoNoduleBlocks = len(huBlocksNoNoduleLung)
    numNoNoduleNoLungBlocks = len(huBlocksNoNoduleNoLung)

    huBlocksNoNoduleLungOut = []
    binBlocksNoNoduleLungOut = []
    huBlocksNoNoduleNoLungOut = []
    binBlocksNoNoduleNoLungOut = []

    for ind1 in getRandomIndices(numNoNoduleBlocks,numNoduleBlocks,minNumSamplesUse):
        huBlocksNoNoduleLungOut.append(huBlocksNoNoduleLung[ind1])
        binBlocksNoNoduleLungOut.append(binBlocksNoNoduleLung[ind1])
    for ind2 in getRandomIndices(numNoNoduleNoLungBlocks,numNoduleBlocks,minNumSamplesUse):
        huBlocksNoNoduleNoLungOut.append(huBlocksNoNoduleNoLung[ind2])
        binBlocksNoNoduleNoLungOut.append(binBlocksNoNoduleNoLung[ind2])

    outFile = '/home/zdestefa/LUNA16/data/DOI_huBlockDataSet/huBlocks_'+patID+'.mat'
    sio.savemat(outFile,{
        "binarySum":binarySumValues,
        "huBlocksNodule":huBlocksNodule,
        "huBlocksNoNoduleLung":huBlocksNoNoduleLungOut,
        "huBlocksNoNoduleNoLung":huBlocksNoNoduleNoLungOut,
        "binBlocksNodule":binBlocksNodule,
        "binBlocksNoNoduleLung":binBlocksNoNoduleLungOut,
        "binBlocksNoNoduleNoLung":binBlocksNoNoduleNoLungOut,
        "numLungPixels":numPossibleLungValues,
        "cancerLikelihoods":cancerLikelihoods})

"""
finalPtsDisplay = np.zeros((len(finalPts),3))
for ii in range(len(finalPts)):
    finalPtsDisplay[ii,:]=finalPts[ii]
print('Checking validity done. Now displaying samples...')
ax.scatter(finalPtsDisplay[:,0],finalPtsDisplay[:,1],finalPtsDisplay[:,2])
plt.show()
"""