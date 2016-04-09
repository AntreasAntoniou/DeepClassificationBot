# -*- coding: utf-8 -*-
import os
import time
from multiprocessing import TimeoutError

import requests
import pytest

import bot
import mocks


def test_fetch_cvimage_from_url(monkeypatch):
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', '1920x1080.png')
    with open(fixture_path, 'rb') as f:
        image = f.read()
    monkeypatch.setattr(requests, 'get', mocks.mock_get(image))
    image = bot.fetch_cvimage_from_url('this url is ignored')
    assert image is not None


def test_fetch_cvimage_from_url_non_image(monkeypatch):
    monkeypatch.setattr(requests, 'get', mocks.mock_get('non-image string'))
    image = bot.fetch_cvimage_from_url('this url is ignored')
    assert image is None


def test_fetch_cvimage_from_url_timeout(monkeypatch):
    long_func = lambda *args, **kwargs: time.sleep(3)
    monkeypatch.setattr(requests, 'get', long_func)
    with pytest.raises(TimeoutError):
        bot.fetch_cvimage_from_url('this url is ignored', timeout_max_timeout=1)


def test_fetch_cvimage_from_url_too_large(monkeypatch):
    monkeypatch.setattr(requests, 'get', mocks.mock_get('12'))
    with pytest.raises(ValueError):
        bot.fetch_cvimage_from_url('this url is ignored', maxsize=1)
