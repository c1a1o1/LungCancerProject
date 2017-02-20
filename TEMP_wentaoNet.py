from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

origNet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
net = Model(input=origNet.input,output=origNet.get_layer('activation_48').output)
net2 = Sequential()
for ii in range(len(net.layers)):
    net2.add(origNet.get_layer(index=ii))

net2.add(Activation('sigmoid'))
net2.add(MaxPooling3D(pool_size=(1,1,25)))