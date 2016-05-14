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


Prediction = namedtuple('Prediction', 'rank category probability')


def load_model(input_shape, n_outputs=100):
    '''Loads and compiles pre-trained model to be used for real-time predictions'''
    model = m.get_deep_anime_model(input_size=input_shape, n_outputs=n_outputs)
    model.load_weights("pre_trained_weights/latest_model_weights.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_data_from_folder(test_image_folder, mean=None, size=256):
    '''Extracts images from image folder and gets them ready for use with the deep neural network.
    Return: [(cvimage, name_string)]
    '''
    image_names = []
    for subdir, _, files in os.walk(test_image_folder):
        for filename in files:
            if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
                continue
            filepath = os.path.join(subdir, filename)
            image, name = get_data_from_file(filepath, size, mean=mean)
            if image is not None:
                image_names.append((image, name))

    return image_names


def get_data_from_file(filepath, size=256, mean=None):
    '''Get image from file ready to be used with the deep neural network.
    Return: (cvimage, name_string)
    '''

    image = cv2.imread(filepath)
    if image is None:
        return None, None

    image = normalize_cvimage(image, size=size, mean=mean)
    name = os.path.basename(filepath)

    return image, name


def normalize_cvimage(cvimage, size=256, mean=None):
    result = data.resize(cvimage, size)
    if mean is not None:
        result = result - mean
    return result / 255


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
        for i, item in enumerate(sample[:top_k]):
            res.append(Prediction(i + 1, categories[item], y[0, item]))
    return res


if __name__ == '__main__':
    import h5py
    # If used as script then run example use case
    import sys
    import urllib
    image_size = 128  # change this to match your image size
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
        image_names = get_data_from_folder(test_image_folder, mean=average_image, size=image_size)
    elif image:
        image_name = get_data_from_file(test_image_path, mean=average_image, size=image_size)
        image_names = [image_name]

    for image, name in image_names:
        y = apply_model(image, model, category_to_catnames)
        print("______________________________________________")
        print("Image Name: {}".format(name))
        print("Categories: ")
        for pred in y:
            print("{0}. {1} {2:.2%}".format(pred.rank, pred.category, pred.probability))
        print("______________________________________________")
