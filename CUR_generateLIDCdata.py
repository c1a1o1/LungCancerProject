import xml.etree.ElementTree as ET
import os
import dicom
import numpy as np
from shutil import copyfile

def get_3d_data(path):
    slices = [dicom.read_file(os.path.join(path,s)) for s in os.listdir(path) if s.endswith(".dcm")]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return [s.SliceLocation for s in slices],np.stack([s.pixel_array for s in slices]),slices[0]


startDir = '/home/zdestefa/LUNA16/data/DOI'

targetDir = '/home/zdestefa/LUNA16/data/DOI_mod'

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
            locData,pixelArray,dcmInfo = get_3d_data(curDir)
            xmlFileDest = os.path.join(targetDir,'xmlFile_'+name+'.xml')
            locFile = os.path.join(targetDir,'sliceLocInfo_'+name+'.npy')
            pixelFile = os.path.join(targetDir,'rawDCM_'+name+'.npy')
            dcmInfoFile = os.path.join(targetDir,'dcmData_'+name+'.npy')
            copyfile(xmlFile,xmlFileDest)
            np.save(locFile,locData)
            np.save(pixelFile,pixelArray)
            np.save(dcmInfoFile,dcmInfo)