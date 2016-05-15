'''
This module provides all the methods needed to succesfully deploy a model. We provide methods for URL based, file based
and folder based deployment. Allowing you to choose one to mix and match with your own project's needs. The main method
shows a use case of the deploy.py module, and also has argument parsing which allows you to quickly test your models.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from collections import namedtuple

import cv2
import numpy as np

import model as m
import data


Prediction = namedtuple('Prediction', 'category probability')


def load_model(input_shape, n_outputs=100):
    '''Loads and compiles pre-trained model to be used for real-time predictions'''
    model = m.get_model(input_size=input_shape, n_outputs=n_outputs)
    model.load_weights("pre_trained_weights/latest_model_weights.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_data_from_folder(test_image_folder, mean=None, size=256):
    '''Extracts images from image folder and gets them ready for use with the deep neural network'''
    # dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    names = []
    for subdir, dir, files in os.walk(test_image_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath)
                if image is not None:
                    image = data.resize(image, size=size)
                    names.append(file)
                    if mean is not None:
                        image = image - mean
                    image = image / 255
                    images.append(image)

    return images, names


def get_data_from_file(filepath, size=256, mean=None):
    '''Get image from file ready to be used with the deep neural network'''

    image = cv2.imread(filepath)
    image = data.resize(image, size)

    if mean is not None:
        image = image - mean

    image = (image) / 255
    bits = filepath.split("/")
    name = bits[-1]

    return image, name


def apply_model(X, model, categories, top_k=3):
    '''Apply model and produce top k predictions for given images
    Returns: [Prediction]
    '''
    X = np.array([X])
    y = model.predict_proba(x=X, batch_size=1)
    top_n = y.argsort()
    res = []
    for sample in top_n:
        sample = sample.tolist()
        sample.reverse()
        for item in sample[:top_k]:
            res.append(Prediction(categories[item], y[0, item]))
    return res


if __name__ == '__main__':
    import h5py
    # If used as script then run example use case
    import sys
    import urllib
    image_size = 128 #change this to match your image size
    test_image_path = ''
    test_image_folder = ''
    image = False
    folder = False
    dataset = h5py.File("data.hdf5", "r")
    average_image = dataset['mean'][:]

    if sys.argv[1] == "--URL":
        link = sys.argv[2]
        bits = link.split("/")
        test_image_path = "downloaded_images/" + str(bits[-1])
        urllib.urlretrieve(link, test_image_path)
        image = True
    elif sys.argv[1] == "--image-path":
        test_image_path = str(sys.argv[2])
        image = True
    elif sys.argv[1] == "--image-folder":
        test_image_folder = sys.argv[2]
        folder = True

    catname_to_categories = data.get_categories()
    category_to_catnames = {v: k for k, v in catname_to_categories.items()}

    # this should be run once and kept in memory for all predictions
    # as re-loading it is very time consuming
    model = load_model(input_shape=image_size, n_outputs=len(category_to_catnames))

    if folder:
        images, names = get_data_from_folder(test_image_folder, mean=average_image, size=image_size)
    elif image:
        image, name = get_data_from_file(test_image_path, mean=average_image, size=image_size)
        images = [image]
        names = [name]

    for image, name in zip(images, names):
        y = apply_model(image, model, category_to_catnames)
        print("______________________________________________")
        print("Image Name: {}".format(name))
        print("Categories: ")
        for i, pred in enumerate(y):
            print("{0}. {1} {2:.2%}".format(i + 1, pred.category, pred.probability))
        print("______________________________________________")
