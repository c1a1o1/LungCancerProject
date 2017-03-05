import xml.etree.ElementTree as ET
import os
import dicom
import numpy as np
import scipy.io as sio
from shutil import copyfile

def get_3d_data(path):
    slices = [dicom.read_file(os.path.join(path,s)) for s in os.listdir(path) if s.endswith(".dcm")]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return [s[0x20, 0x32].value[2] for s in slices]


startDir = '/home/zdestefa/LUNA16/data/DOI'

targetDir = '/home/zdestefa/LUNA16/data/DOI_modSliceLoc'

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
            locData = get_3d_data(curDir)
            locFile = os.path.join(targetDir,'sliceLocData_'+name+'.mat')
            sio.savemat(locFile,{"locData":locData})