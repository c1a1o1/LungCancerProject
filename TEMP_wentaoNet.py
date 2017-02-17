from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

origNet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
net = Model(input=origNet.input,output=origNet.get_layer('activation_49').output)
net.add(Activation('sigmoid'))