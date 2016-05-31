'''This module provides all the methods needed for data extraction, preprocessing, storing and retrieval.

A deep neural network is nothing but a bunch of random parameters without massive amounts of high quality data
to train it on, and as such a large percentage of project time was spent building the data.py methods. Our main
storage system is HDF5 as it provides a pythonic object oriented approach to storing along with numpy combability and
very fast streaming capabilities. An additional plus was the ability of "fancy indexing" or list of indexes type of
sample access. Most of the details are provided in the methods docstrings
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import pickle
import random

import cv2
import numpy as np
import h5py
from keras.utils import np_utils


def extract_data(rootdir=None, size=256):
    '''Extracts the data from the downloaded_images folders
        Attributes:
            size: The size to which to resize the images. All images must be the same size so that
            they can be trained using a Deep CNN.
            e.g size=256, images will be 256x256
        Returns a list(X) with the images and their labels(y) in string form, e.g.('dog')
    '''

    X = []
    y = []
    if rootdir is not None:
        search_folder = rootdir
    else:
        search_folder = os.path.dirname(os.path.abspath(__file__)) + "/downloaded_images/"

    count = 0

    for subdir, dir, files in os.walk(search_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                bits = subdir.split("/")
                category = bits[-1]
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath)
                try:
                    image = resize(image, size=size)
                    if image is not None:
                        X.append(image)
                        y.append(category)
                    count = count + 1
                    print("Images proccesed {0}".format(count))
                except:
                    pass

    return X, y


def preprocess_data(X, y, save=True, preset=None, subtract_mean=True):
    '''Preprocesses the data that have already been extracted with extract_data() method
       Attributes:
           X: A list containing images of shape (3, width, height)
           y: A list containing the image labels in string form
        Returns data in numpy form ready to be used for training
    '''

    X = np.array(X)
    categories = set()

    for label in y:  # get all the unique categories
        categories.add(label)

    # build a dictionary that maps categories in string form to a unique id
    categories = dict(zip(categories, list(range(len(categories)))))

    y_temp = []
    for label in y:  # encode y with the unique ids
        y_temp.append(categories[label])

    y_temp = np.array(y_temp)
    X = X.astype(np.float32)
    mean = X.mean(axis=0)  # get mean
    if subtract_mean:
        X = X - mean

    # save categories for future use
    pickle.dump(categories, open("data/categories.p", "wb"))

    y = np_utils.to_categorical(y_temp, max(y_temp) + 1)
    if preset is not None:
        y = np_utils.to_categorical(y_temp, preset)

    n_samples = y.shape[0]
    n_categories = len(categories)

    if save:
        h5f = h5py.File('data/data.hdf5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
        h5f.create_dataset('nb_samples', data=n_samples)
        h5f.create_dataset('n_categories', data=n_categories)
        h5f.create_dataset('mean', data=mean)
        h5f.close()

    return X, y, n_samples, len(categories)


def get_mean(dataset_path='data/data.hf5'):
    try:
        h5f = h5py.File(dataset_path, 'r')
        return h5f['mean'][:]
    except IOError:
        return np.load("data/mean.npy")


def get_categories():
    '''Load categories names'''

    categories = pickle.load(open("data/categories.p", "rb"))
    return categories


def produce_train_indices(dataset_indx, number_of_samples, val_indx):
    dataset_indx = np.delete(dataset_indx, val_indx)
    np.random.seed(seed=None)
    train = np.random.choice(dataset_indx, size=number_of_samples)
    train = np.unique(train)
    return (np.sort(train)).tolist()


def produce_validation_indices(dataset_indx, number_of_samples):
    np.random.seed(2048)
    val = np.random.choice(dataset_indx, size=number_of_samples)
    val = np.sort(val)
    val = np.unique(val)
    return val.tolist()


def load_dataset_bit_from_hdf5(train_indices, val_indices, only_train=True):
    if only_train:
        h5f = h5py.File('data.hdf5', 'r')
        X_train = h5f['X'][train_indices]
        y_train = h5f['y'][train_indices]
        h5f.close()
        return X_train, y_train
    else:
        h5f = h5py.File('data.hdf5', 'r')
        X_train = h5f['X'][train_indices]
        y_train = h5f['y'][train_indices]
        X_val = h5f['X'][val_indices]
        y_val = h5f['y'][val_indices]
        h5f.close()
        return X_train, y_train, X_val, y_val


def augment_data(X_train, random_angle_max=180, mirroring_probability=0.5):
    '''Augment data with random mirrors and random rotations'''
    for i in range(len(X_train)):
        random_angle = random.randint(0, random_angle_max)
        mirror_decision = random.randint(0, 100)
        flip_orientation = random.randint(0, 1)
        for channel in range(len(X_train[i])):
            rows, cols = X_train[i, channel].shape
            M = cv2.getRotationMatrix2D((cols // 2, rows // 2), random_angle, 1)
            rotated_image = cv2.warpAffine(X_train[i, channel], M, (cols, rows))
            if mirror_decision < mirroring_probability * 100:
                X_train[i, channel] = cv2.flip(rotated_image, flipCode=flip_orientation)
            else:
                X_train[i, channel] = rotated_image
    return X_train


def resize(img, size):
    """resize image into size x size
        Attributes:
            img: The image to resize
            size: The size to resize the image
    """
    print("preresize")
    img = cv2.resize(img, (size, size))
    print("preroll")
    img = np.rollaxis(img, 2)
    return img


def split_data(X, y, split_ratio=0.5):
    '''Splits data into training and testing-sets'''
    random.seed(4096)
    train_idx = []
    test_idx = []
    for i in range(X.shape[0]):
        decision = random.randint(0, 99)
        if decision < (100 - (split_ratio * 100)):
            train_idx.append(i)
        else:
            test_idx.append(i)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test
