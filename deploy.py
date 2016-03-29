import model as m
import os
import data
import cv2
import numpy as np
'''
This module provides all the methods needed to succesfully deploy a model. We provide methods for URL based, file based
and folder based deployment. Allowing you to choose one to mix and match with your own project's needs. The main method
shows a use case of the deploy.py module, and also has argument parsing which allows you to quickly test your models.
'''


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

    return np.array(images), names



def get_data_from_file(filepath, size=256, mean=None):
    '''Get image from file ready to be used with the deep neural network'''
    import data

    image = cv2.imread(filepath)
    image = data.resize(image, size)

    if mean is not None:
        image = image - mean

    image = (image) / 255
    bits = filepath.split("/")
    name = bits[-1]

    return image, name


def apply_model(X, model, categories, multi=False, top_k=3):
    '''Apply model and produce top k predictions for given images'''
    y = []
    if not multi:
        X = np.array([X])
        y_temp = model.predict_proba(X=X, batch_size=1)
        top_n = y_temp.argsort()
        res = []
        for sample in top_n:
            sample = sample.tolist()
            sample.reverse()
            for item in sample:
                res.append(str(categories[item]) + ": " + str(y_temp[0, item]))
        return res
    else:
        for image in X:
            image = np.array([image])
            y_temp = model.predict_proba(X=image, batch_size=1)
            top_n = y_temp.argsort()
            res = []
            for sample in top_n:
                sample = sample.tolist()
                sample.reverse()
                for item in range(top_k):
                    res.append(str(categories[sample[item]]) + ": " + str(y_temp[0, sample[item]]))
            y.append(res)
    return y


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
    n_categories = dataset['n_categories'].value
    average_image = dataset['mean'][:]

    if sys.argv[1] == "--URL":
        link = sys.argv[2]
        bits = link.split("/")
        test_image_path = "downloaded_images/" + str(bits[-1])
        urllib.urlretrieve(link, test_image_path)
        image = True
    elif sys.argv[1] == "--image_path":
        test_image_path = str(sys.argv[2])
        image = True
    elif sys.argv[1] == "--image_folder":
        test_image_folder = sys.argv[2]
        folder = True

    categories = data.get_categories()
    categories_to_strings = dict()

    for key in categories.iterkeys():
        categories_to_strings[categories[key]] = key

    model = load_model(input_shape=image_size, n_outputs=n_categories)
    # this should be run once and kept in memory for all predictions
    # as re-loading it is very time consuming

    if folder:
        images, names = get_data_from_folder(test_image_folder, mean=average_image, size=image_size)
        y = apply_model(images, model, categories_to_strings, multi=True)
        print(len(y))
        for i in range(len(y)):
            item = y[i]
            print("______________________________________________")
            print("Image Name: {}".format(str(names[i])))
            print(" Categories: ")
            for j in range(5):
                print("{0}. {1}".format(j + 1, item[j] * 100))
            print("______________________________________________")
    elif image:
        images, name = get_data_from_file(test_image_path, mean=average_image, size=image_size)
        y = apply_model(images, model, categories_to_strings, multi=False)
        print("_________________________________________________")
        print("Image Name: {}".format(name))
        print("Categories: ")
        for i in range(5):
            print("{0}. {1}".format(i + 1, y[i]))
        print("_________________________________________________")
