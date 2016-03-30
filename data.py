import pickle
import random
import os
import cv2
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as p


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
                #print(category)
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath)
                if image is not None:
                    image = resize(image, size=size)
                    X.append(image)
                    y.append(category)
                print(count)
                count = count + 1
    return X, y


def preprocess_data(X, y, save=True, preset=None):
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

    categories = dict(zip(categories, range(len(categories))))  # build a dictionary that maps categories in string form
                                                                # to a unique id
    y_temp = []
    for label in y:  # encode y with the unique ids
        y_temp.append(categories[label])
    #print(y_temp)
    y_temp = np.array(y_temp)
    X = X.astype(np.float32)
    mean = X.mean(axis=0)  # get mean
    X = X - X.mean(axis=0)
    print(y_temp)
    # save mean
    np.save("data/mean.npy", mean)

    # save categories for future use
    pickle.dump(categories, open("categories.p", "wb"))

    y = np_utils.to_categorical(y_temp, max(y_temp)+1)
    if preset is not None:
        y = np_utils.to_categorical(y_temp, preset)

    print(X.shape)
    print(y.shape)
    if save:
        np.save("data/X.npy", X)
        np.save("data/y.npy", y)

    return X, y


def get_metadata():
    '''Load metadata'''

    #fo = open("mean.txt", 'r')
    #lines = fo.readlines()
    #line = lines[0]
    #bits = line.split(",")
    #mean = float(bits[-1])
    categories = pickle.load(open("categories.p", "rb"))

    return categories

def load_data():

    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = (X) / (255)
    X_res = np.zeros((X.shape[0], X.shape[1], 128, 128))
    for i in range(len(X)):
        for channel in range(len(X[i])):
            X_res[i, channel]= cv2.resize(X[i, channel], dsize=(128, 128))
            #p.imshow(X_res[i, channel])
            #p.show()
    X = np.array(X_res)
    return X, y

def augment_data(X_train, random_angle_max=360, mirroring_probability=0.5):
    for i in range(len(X_train)):
        random_angle = random.randint(0, random_angle_max)
        mirror_decision = random.randint(0, 100)
        flip_orientation = random.randint(0, 1)
        for channel in range(len(X_train[i])):
            rows,cols = X_train[i, channel].shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),random_angle,1)
            rotated_image = cv2.warpAffine(X_train[i, channel],M,(cols,rows))
            if mirror_decision<mirroring_probability*100:
                X_train[i, channel] = cv2.flip(rotated_image, flipCode=flip_orientation)
            else:
                X_train[i, channel] = rotated_image
            #print(X_train[i, channel])
            #p.imshow(X_train[i, channel])
            #p.show()
    return X_train


def resize(img, size):
    """resize image into size x size
        Attributes:
            img: The image to resize
            size: The size to resize the image
    """

    img = cv2.resize(img, (size, size))
    img = np.rollaxis(img, 2)
    return img


def split_data(X, y, split_ratio=0.1):
    ''''''
    random.seed(4096)
    train_idx = []
    test_idx = []
    for i in range(X.shape[0]):
        decision = random.randint(0, 99)
        if decision < (100-(split_ratio*100)):
            train_idx.append(i)
        else:
            test_idx.append(i)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test