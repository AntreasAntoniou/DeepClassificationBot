from keras.models import Sequential
import model as m
import os
import data
import cv2
import numpy as np


def load_model():
    '''Loads and compiles pre-trained model to be used for real-time predictions'''
    model = m.get_model()
    model.load_weights("model_weights.hdf5")
    model.compile(loss='crossentropy_classification_error', optimizer='adam')
    return model

def get_data_from_folder(test_image_folder):
    '''Extracts images from image folder and gets them ready for use with the deep neural network'''
    categories = data.get_metadata()
    dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    for subdir, dir, files in os.walk(dir+"/"+test_image_folder+"/"):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            image = data.resize(image)
            image = image.astype(np.float32)
            image = (image) / 255
            images.append(image)

def get_data_from_file(filepath):
    '''Get image from file ready to be used with the deep neural network'''
    categories = data.get_metadata()
    image = cv2.imread(filepath)
    image = (image) / 255

    return image

def apply_model(X, model, categories):
    '''Apply model and produce top 3 predictions for given images'''
    model = Sequential()
    y = []
    for image in X:
        y_temp = model.predict_proba(X=image, batch_size=1)
        top_n = y_temp.argsort()[-3:][::-1]
        res = []
        for item in top_n:
            res.append(categories[item])
        y.append(res)
    return y

if __name__ == '__main__':
    #If used as script then run example use case
    test_image_path = ''
    test_image_folder = ''

    model = load_model()

    categories = data.get_metadata()

    #images = get_data_from_folder(test_image_folder)
    images = get_data_from_file(test_image_path)

    apply_model(images, model, categories)