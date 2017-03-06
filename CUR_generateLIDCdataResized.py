import xml.etree.ElementTree as ET
import os
import dicom
import numpy as np
from shutil import copyfile
from scipy.ndimage.interpolation import zoom
import scipy.io as sio


def get_3d_data(path):
    slices = [dicom.read_file(os.path.join(path,s)) for s in os.listdir(path) if s.endswith(".dcm")]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    slice1 = slices[0]
    rawVolume = np.zeros((slice1.Rows,slice1.Columns,len(slices)))
    for jj in range(len(slices)):
        rawVolume[:,:,jj] = slices[jj].pixel_array
    volumeHU = rawVolume*slice1.RescaleSlope + slice1.RescaleIntercept
    spacingXY = slice1.PixelSpacing
    spacingZ = slice1.SliceThickness
    spacingXYZ = [spacingXY[0],spacingXY[1],spacingZ]
    return volumeHU,spacingXYZ
    #return [s.SliceLocation for s in slices],np.stack([s.pixel_array for s in slices]),slices[0]


startDir = '/home/zdestefa/LUNA16/data/DOI'

targetDir = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfo'
targetDir2 = '/home/zdestefa/LUNA16/data/DOI_huAndResizeInfoMatlab'

cases = os.listdir(startDir)

numPt = 0

for root,dirs,files in os.walk(startDir):
    for name in dirs:
        curDir = os.path.join(root,name)
        files = os.listdir(curDir)
        curNumDcm=0
        curNumXML=0
        xmlFile = ""
        for file in files:
            if file.endswith(".dcm"):
                curNumDcm = curNumDcm+1
            if file.endswith(".xml"):
                curNumXML=curNumXML+1
                xmlFile = os.path.join(curDir,file)
        if curNumDcm>10 and curNumXML>0:
            print('Now processing patient ' + str(numPt))
            numPt = numPt+1
            pixelArray,resizeArray = get_3d_data(curDir)
            #pixelFile = os.path.join(targetDir2,'HUarray_'+name+'.npy')
            #resizeFile = os.path.join(targetDir2, 'resizeTuple_' + name + '.npy')
            outputMATfile = os.path.join(targetDir2, 'HUarrayResizeInfo_' + name + '.mat')
            #np.save(pixelFile,pixelArray)
            #np.save(resizeFile,resizeArray)
            sio.savemat(outputMATfile,{"pixelArray":pixelArray,"resizeArray":resizeArray})

