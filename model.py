from __future__ import absolute_import
from __future__ import print_function

from keras.layers.recurrent import LSTM
from keras.models import Sequential, Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

'''
This module provides all the methods needed to build a deep neural network using the Keras deep learning library.
Keras was build by Francois Chollet (fchollet) and is essentially an abstraction library that is universal in the sense
that it can run on top of both theano and tensorflow which makes it especially powerful and adaptive to your project's
needs.
'''


def get_model(n_outputs=1000, input_size=256):
    '''Builds a Deep Convolutional Neural Network of architecture VGG-Net as described in
       paper http://arxiv.org/pdf/1409.1556.pdf and adapted with batch_norm and dropout regularization
       Returns the model ready for compilation and training or predictions
       we have commented out dropout in between the conv layers because it was not needed for our use cases. However if
       you find that your models overfit you can choose to uncomment and add them. Back into your architecture.
    '''
    conv = Sequential()

    conv.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3, input_size, input_size)))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(64, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    #conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    #conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    #conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    #conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())

    conv.add(Flatten())
    conv.add(Dropout(0.5))
    conv.add(Dense(4096))
    conv.add(BatchNormalization())
    conv.add(Dropout(0.5))
    conv.add(Dense(4096))
    conv.add(BatchNormalization())
    conv.add(Dropout(0.5))
    conv.add(Dense(n_outputs))
    conv.add(Activation('softmax'))
    print(conv.summary())
    return conv


def get_deep_anime_model(n_outputs=1000, input_size=128):
    '''The deep neural network used for deep anime bot'''
    conv = Sequential()

    conv.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(3, input_size, input_size)))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(64, 3, 3, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 3, 3, activation='relu'))
    #conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(128, 1, 1, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 3, 3, activation='relu'))
    #conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(256, 1, 1, activation='relu'))
    conv.add(MaxPooling2D((2, 2), strides=(2, 2)))
    conv.add(BatchNormalization())
    #conv.add(Dropout(0.5))

    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 3, 3, activation='relu'))
    #conv.add(ZeroPadding2D((1, 1)))
    conv.add(Convolution2D(512, 1, 1, activation='relu'))
    conv.add(AveragePooling2D((8, 8), strides=(2, 2)))
    conv.add(BatchNormalization())
    # conv.add(Dropout(0.5))

    # conv.add(ZeroPadding2D((1, 1)))
    # conv.add(Convolution2D(512, 3, 3, activation='relu'))
    # conv.add(ZeroPadding2D((1, 1)))
    # conv.add(Convolution2D(512, 3, 3, activation='relu'))
    # #conv.add(ZeroPadding2D((1, 1)))
    # conv.add(Convolution2D(512, 1, 1, activation='relu'))
    # conv.add(AveragePooling2D((4, 4)))

    #conv.add(BatchNormalization())
    conv.add(Flatten())
    conv.add(Dropout(0.5))
    conv.add(Dense(2048))
    conv.add(BatchNormalization())
    conv.add(Dropout(0.7))
    conv.add(Dense(2048))
    conv.add(BatchNormalization())
    conv.add(Dropout(0.7))
    conv.add(Dense(n_outputs))
    conv.add(Activation('softmax'))
    print(conv.summary())
    return conv
