# -*- coding: utf-8 -*-

import cv2
import numpy as np
import h5py
import requests

import data
import deploy

from .decorators import timeout
from . import exceptions as exc
from .shortcuts import at_random


@timeout(30)
def fetch_cvimage_from_url(url, maxsize=10 * 1024 * 1024):
    req = requests.get(url, timeout=5, stream=True)
    content = ''
    for chunk in req.iter_content(2048):
        content += chunk
        if len(content) > maxsize:
            req.close()
            raise ValueError('Response too large')
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    cv2_img_flag = cv2.CV_LOAD_IMAGE_COLOR
    image = cv2.imdecode(img_array, cv2_img_flag)
    return image


class MockClassifier(object):
    def classify(self, *args, **kwargs):
        message = at_random(
            "I hope a mock message like this won't get caught by Twitter's spam filter",
            "But I must explain to you how all this mistaken idea was born",
            "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis",
            "Excepteur sint occaecat cupidatat non proident",
        )
        return [deploy.Prediction(1, message, 100)]


class ImageClassifier(object):
    def __init__(self, dataset_path, input_shape, model_name='model'):
        catname_to_categories = data.get_categories()
        self.category_to_catnames = {v: k for k, v in catname_to_categories.items()}
        self.model = deploy.load_model(
            input_shape=input_shape,
            n_outputs=len(catname_to_categories),
            model_name=model_name)
        self.input_shape = input_shape
        self.average_image = data.get_mean(dataset_path)

    def classify(self, cvimage):
        normalized = deploy.normalize_cvimage(cvimage, size=self.input_shape, mean=self.average_image)
        return deploy.apply_model(normalized, self.model, self.category_to_catnames)


class URLClassifier(object):
    def __init__(self, image_classifier):
        self._image_classifier = image_classifier

    def classify(self, url=None):
        cvimage = fetch_cvimage_from_url(url)

        if cvimage is None:
            raise exc.NotImage(url)

        return self._image_classifier.classify(cvimage)


class RemoteClassifier(object):
    def __init__(self, base_url):
        self._base_url = base_url

    def classify(self, **params):
        try:
            r = requests.get(self._base_url, params=params, timeout=60).json()
            if 'error' in r:
                raise exc.RemoteError(r['error'])
            return map(lambda guess: deploy.Prediction(**guess), r['y'])
        except requests.exceptions.Timeout:
            raise exc.TimeoutError
