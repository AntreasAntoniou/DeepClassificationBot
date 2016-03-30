from keras.models import Sequential
import model as m
import os
import data
import cv2
import numpy as np


def load_model():
    '''Loads and compiles pre-trained model to be used for real-time predictions'''
    model = m.get_model(50)
    model.load_weights("pre_trained_weights/model_weights.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def get_data_from_folder(test_image_folder):
    '''Extracts images from image folder and gets them ready for use with the deep neural network'''
    categories = data.get_metadata()
    #dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    mean = np.load("mean.npy")
    for subdir, dir, files in os.walk(test_image_folder):
        for file in files:
             if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                bits = subdir.split("/")
                filepath = os.path.join(subdir, file)
                image = cv2.imread(filepath)
                if image is not None:
                    image = image - mean
                    image = image/255
                    image = data.resize(image, size=128)
                    images.append(image)
                    print(len(images))
    return np.array(images)

def get_data_from_file(filepath):
    '''Get image from file ready to be used with the deep neural network'''
    import data

    image = cv2.imread(filepath)
    print(image.shape)
    image = (image) / 255
    image = data.resize(image, 128)
    print(image.shape)

    return image

def apply_model(X, model, categories, multi=False):
    '''Apply model and produce top 3 predictions for given images'''
    y = []
    if multi==False:
        print("Got in image predictions")
        print(X.shape)
        X = np.array([X])
        print(X.shape)
        y_temp = model.predict_proba(X=X, batch_size=1)
        print(y_temp)
        top_n = y_temp.argsort()
        res = []
        for sample in top_n:
            print(sample)
            sample = sample.tolist()
            sample.reverse()
            print(sample)
            for item in sample:
                res.append(str(categories[item])+": "+str(y_temp[0, item]))
        return res
    elif multi:
        for image in X:
            print("Got in image predictions")
            print(X.shape)
            image = np.array([image])
            print(X.shape)
            y_temp = model.predict_proba(X=image, batch_size=1)
            print(y_temp)
            top_n = y_temp.argsort()
            res = []
            for sample in top_n:
                print(sample)
                sample = sample.tolist()
                sample.reverse()
                print(sample)
                for item in range(5):
                    res.append(str(categories[sample[item]])+": "+str(y_temp[0, sample[item]]))
            y.append(res)
    return y

if __name__ == '__main__':
    #If used as script then run example use case
    import sys
    import urllib
    test_image_path = ''
    test_image_folder = ''
    image = False
    folder = False
    print(sys.argv)
    if sys.argv[1]== "--URL":
        link = sys.argv[2]
        bits = link.split("/")
        test_image_path = "downloaded_images/"+str(bits[-1])
        urllib.urlretrieve(link, test_image_path)
        image=True
    elif sys.argv[1] == "--image_path":
        test_image_path = str(sys.argv[2])
        image = True
    elif sys.argv[1] == "--image_folder":
        test_image_folder = sys.argv[2]
        folder = True


    model = load_model()

    categories = data.get_metadata()
    categories_to_strings = dict()

    for key in categories.iterkeys():
        categories_to_strings[categories[key]] = key

    print(categories)
    print(categories_to_strings)
    if folder:
        images = get_data_from_folder(test_image_folder)
        y = apply_model(images, model, categories_to_strings, multi=True)
        print(len(y))
        for item in y:
            for i in range(1):
                print(item[i]+"\n")
    elif image:
        images = get_data_from_file(test_image_path)
        y = apply_model(images, model, categories_to_strings, multi=False)
        for i in range(1):
            print(y[i]+"\n")
