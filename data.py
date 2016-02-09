import pickle
import random
import os
import cv2
import numpy as np
from keras.utils import np_utils


def extract_data(size=256):
    '''Extracts the data from the downloaded_images folders
        Attributes:
            size: The size to which to resize the images. All images must be the same size so that
            they can be trained using a Deep CNN.
            e.g size=256, images will be 256x256
        Returns a list(X) with the images and their labels(y) in string form, e.g.('dog')
    '''

    X = []
    y = []
    dir = os.path.dirname(os.path.abspath(__file__))

    for subdir, dir, files in os.walk(dir+"/downloaded_images/"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                bits = subdir.split("/")
                category = bits[-1]
                print(category)
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath)
                if image is not None:
                    image = resize(image, size=size)
                    X.append(image)
                    y.append(category)

    return X, y


def preprocess_data(X, y, save=True):
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

    y_temp = np.array(y_temp)
    X = X.astype(np.float32)
    mean = X.mean()  # get mean

    # save mean
    fo = open("mean.txt", 'wb+')
    output = "mean,"+str(mean)+"\n"
    fo.write(output)
    fo.close()

    # save categories for future use
    pickle.dump(categories, open("categories.p", "wb"))

    X = (X-mean) / 255  # Scale by 255 to get the inputs in the range 0 - 1 so that the CNN can understand them
    np_utils.to_categorical(y_temp, 1000)
    print(X.shape)
    print(y_temp.shape)
    if save:
        np.save("data/X.npy", X)
        np.save("data/y.npy", y)



def get_metadata():
    '''Load metadata'''

    fo = open("mean.txt", 'r')
    lines = fo.readlines()
    line = lines[0]
    bits = line.split(",")
    mean = float(bits[-1])
    categories = pickle.load(open("categories.p", "rb"))

    return mean, categories

def load_data():

    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    return X, y

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
            test_idx.append(i)
        else:
            train_idx.append(i)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test