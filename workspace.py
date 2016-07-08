# -*- coding: utf-8 -*-
'''
Workspace knows where to store what.
'''
import os
from collections import namedtuple


class Workspace(namedtuple('Workspace', 'home')):
    @property
    def raw_images_dir(self):
        return os.path.join(self.home, 'downloaded_images')

    @property
    def categories_path(self):
        return os.path.join(self.home, 'data', 'categories.p')

    @property
    def data_path(self):
        return os.path.join(self.home, 'data', 'data.hdf5')

    @property
    def model_weights_path(self):
        return os.path.join(self.home, 'pre_trained_weights', 'model_weights.hdf5')

    @property
    def latest_model_weights_path(self):
        return os.path.join(self.home, 'pre_trained_weights', 'latest_model_weights.hdf5')

    @property
    def mean_path(self):
        return os.path.join(self.home, 'data', 'mean.npy')


DEFAULT_WORKSPACE = Workspace(home=os.path.join(os.getcwd(), 'default_workspace'))
