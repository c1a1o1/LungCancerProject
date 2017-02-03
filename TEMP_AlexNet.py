

from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio
from scipy.misc import imread, imresize, imsave
import time
import datetime
import csv
from convnetskeras.convnets import preprocess_image_batch, convnet
from sklearn.cross_validation import train_test_split
from sklearn import random_projection
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids

l2factor = 0#1e-5
l1factor = 0#2e-7
mil=True

alexmodel = convnet('alexnet', weights_path='alexnet_weights.h5', heatmap=False)
#alexmodel = convnet('alexnet', heatmap=False)
model = convnet('alexnet')
for layer, mylayer in zip(alexmodel.layers, model.layers):
  print(layer.name)
  #if mylayer.name == 'mil_1':
  if mylayer.name == 'flatten':
    break
  else:
    weightsval = layer.get_weights()
    print(len(weightsval))
    mylayer.set_weights(weightsval)
