# -*- coding: utf-8 -*-
'''
Workspace knows where to store what.
'''
from collections import namedtuple


class Workspace(namedtuple('Workspace', 'home')):
    pass
