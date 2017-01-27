
import scipy.io as sio
import time
import datetime
import numpy as np

randArray = np.random.rand(3,3)

ts = time.time()

st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')

fileName = 'randFiles/randomArrayFrom_'+st+'.mat'
sio.savemat(fileName,mdict={'randArray':randArray})
