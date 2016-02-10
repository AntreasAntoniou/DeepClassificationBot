from __future__ import print_function

from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def get_model(n_outputs=1000):
    '''Builds a Deep Convolutional Neural Network of architecture VGG-Net as described in
       paper http://arxiv.org/pdf/1409.1556.pdf
       Returns the model ready for compilation and training or predictions
    '''
    conv = Sequential()

    conv.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3, 256, 256)))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(64, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    conv.add(Dropout(0.5))
    conv.add(Dense(4096))
    conv.add(Dropout(0.5))
    conv.add(Dense(n_outputs))
    conv.add(Activation('softmax'))
    print(conv.summary())
    return conv
