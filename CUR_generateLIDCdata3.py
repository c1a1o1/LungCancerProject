
import os

import numpy as np
import scipy.io as sio


startDir = '/home/zdestefa/LUNA16/data/DOI_mod'

targetDir = '/home/zdestefa/LUNA16/data/DOI_modMatlab2'

xmlFiles = [file for file in os.listdir(startDir) if file.startswith("dcmData_")]

for file in xmlFiles:
    targetFile = os.path.join(targetDir,file)
    dcmArray = np.load(os.path.join(startDir,file))
    sio.savemat(targetFile,{"dcmArray":dcmArray})
