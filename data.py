import pickle
import random
import os
import cv2
import numpy as np
from keras.utils import np_utils


def extract_data():
    X = []
    y = []
    dir = os.path.dirname(os.path.abspath(__file__))

    for subdir, dir, files in os.walk(dir+"/downloaded_images/"):
        for file in files:
            bits = subdir.split("/")
            category = bits[-1]
            print(category)
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            image = resize(image, size=256)
            X.append(image)
            y.append(category)

    return X, y

def preprocess_data(X, y):

    X = np.array(X)
    categories = set()

    for label in y:
        categories.add(label)

    categories = dict(zip(categories, range(len(categories))))
    y_temp = []
    for label in y:
        y_temp.append(categories[label])

    y_temp = np.array(y_temp)
    X = X.astype(np.float32)
    mean = X.mean()

    fo=open("mean.txt", 'wb+')
    output="mean,"+mean+"\n"
    fo.write(output)
    fo.close()

    pickle.dump(categories, open("categories.p", "wb"))


    X = (X-mean)/255
    np_utils.to_categorical(y, 1000)
    return X, y_temp

def get_metadata():

    fo = open("mean.txt", 'r')
    lines = fo.readlines()
    line = lines[0]
    bits = line.split(",")
    mean = float(bits[-1])
    categories = pickle.load(open("categories.p", "rb"))

    return mean

def resize(img, size):
       """resize"""

       img=cv2.resize(img, (size, size))

       return img


def split_data(X, y, split_ratio=0.1):
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