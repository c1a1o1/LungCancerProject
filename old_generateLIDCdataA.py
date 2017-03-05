import xml.etree.ElementTree as ET
import os
import scipy.io as sio

startDir = '/home/zdestefa/LUNA16/data/DOI'

dirsToCheck = []

for root,dirs,files in os.walk(startDir):
    for file in files:
        if(file.endswith('.dcm')):
            dirsToCheck.append(root)
            break

sio.savemat('TEMP_dirsCheck.mat',{"dirsToCheck":dirsToCheck})
