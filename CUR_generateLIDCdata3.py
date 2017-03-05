
import os

import numpy as np
import scipy.io as sio


startDir = '/home/zdestefa/LUNA16/data/DOI_mod'

targetDir = '/home/zdestefa/LUNA16/data/DOI_modMatlab2'

fileNum=1
for file in os.listdir(startDir):
    if(file.startswith('dcmData_')):
        print('Processing File ' + str(fileNum))
        fileNum=fileNum+1
        targetFile = os.path.join(targetDir, file)
        fileToLoad = os.path.join(startDir, file)
        dcmArray = np.load(fileToLoad)
        sio.savemat(targetFile, {"dcmArray": dcmArray})