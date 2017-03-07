import numpy as np
import os
import scipy.io as sio
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#curDir = '/home/zdestefa/LUNA16/data/DOI_modNodule'
curDir = 'D:\dev\git\LungCancerProject\DOI_modNodule'

print('Loading Binary Array')

matFiles = os.listdir(curDir)

curFile = os.path.join(curDir,matFiles[0])
binaryArray = sio.loadmat(curFile)
binArray = binaryArray['finalOutputSparse'][0]

print('Binary Array Loaded. Converting to Matrix...')

binArrayDense = np.zeros((512,512,len(binArray)))
for ii in range(len(binArray)):
    binArrayDense[:,:,ii]=binArray[ii].todense()


#samples random points to use as center of 64x64x64 blocks
#make sure they are at least 24 pixels apart
#   this allows some overlap but not too much overlap

numSampledPts = 4500
blockDim = 64
numInXrange = 512-(blockDim+2)
numInYrange = 512-(blockDim+2)
numInZrange = len(binArray)-(blockDim+2)
matrixToMultBy = np.matlib.repmat([numInXrange,numInYrange,numInZrange],numSampledPts,1)

print('Matrix Conversion done. Doing Sampling...')

initSamples = np.random.rand(numSampledPts,3)
multipliedSamples = np.multiply(matrixToMultBy,initSamples)
finalSamples = np.floor(np.add(multipliedSamples,blockDim/2 + 2))

print('Sampling Finished. Now checking validity of each sample')

prismValid = np.ones((512,512,len(binArray)))
finalPts = []
for ii in range(numSampledPts):
    curPt = finalSamples[ii,:]
    curPtR = int(curPt[0])
    curPtC = int(curPt[1])
    curPtS = int(curPt[2])
    if(prismValid[curPtR,curPtC,curPtS]>0):
        finalPts.append(curPt)
        xMin = int(curPtR-blockDim/2)
        xMax = xMin+blockDim
        yMin = int(curPtC-blockDim/2)
        yMax = yMin + blockDim
        zMin = int(curPtS-blockDim/2)
        zMax = zMin + blockDim
        prismValid[xMin:xMax,yMin:yMax,zMin:zMax] = 0

finalPtsDisplay = np.zeros((len(finalPts),3))
for ii in range(len(finalPts)):
    finalPtsDisplay[ii,:]=finalPts[ii]

print('Checking validity done. Now displaying samples...')

ax.scatter(finalPtsDisplay[:,0],finalPtsDisplay[:,1],finalPtsDisplay[:,2])

plt.show()