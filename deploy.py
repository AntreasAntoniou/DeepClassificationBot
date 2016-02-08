from keras.models import Sequential
import model as m
import os
import data
import cv2

def load_model():
    model = m.get_model()
    model.load_weights("model_weights.hdf5")
    model.compile(loss='crossentropy_classification_error', optimizer='adam')
    return model

def get_data_from_folder(test_image_folder):
    mean = data.get_metadata()
    dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    for subdir, dir, files in os.walk(dir+"/"+test_image_folder+"/"):
        for file in files:
            filepath=os.path.join(subdir, file)
            image = cv2.imread(filepath)
            image = (image - mean) / 255
            images.append(image)

def get_data_from_file(filepath):
    mean = data.get_metadata()
    image = cv2.imread(filepath)
    image = (image - mean) / 255

    return image

def apply_model(X, model, categories):
    model = Sequential()
    y = []
    for image in X:
        y_temp = model.predict_proba(X=X, batch_size=128)
        top_n = y_temp.argsort()[-3:][::-1]
        res = []
        for item in top_n:
            res.append(categories[item])
        y.append(res)
    return y

if __name__ == '__main__':
    test_image_path = ''

    model = load_model()

    mean, categories = data.get_metadata()

    images = get_data_from_file(test_image_path)

    apply_model(images, model, categories)