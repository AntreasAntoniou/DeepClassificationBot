# -*- coding: utf-8 -*-
import os
import time
from multiprocessing import TimeoutError

import requests
import pytest

import bot
import deploy
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
    def long_func(*args, **kwargs):
        time.sleep(3)
    monkeypatch.setattr(requests, 'get', long_func)
    with pytest.raises(TimeoutError):
        bot.fetch_cvimage_from_url('this url is ignored', timeout_max_timeout=1)


def test_fetch_cvimage_from_url_too_large(monkeypatch):
    monkeypatch.setattr(requests, 'get', mocks.mock_get('12'))
    with pytest.raises(ValueError):
        bot.fetch_cvimage_from_url('this url is ignored', maxsize=1)


def test_my_guess_honors_max_length_by_truncating_longest():
    y = [
        deploy.Prediction(2, '567890', 0.024),
        deploy.Prediction(7, '012', 0.046),
    ]
    # before truncation:
    # '\n2. 567890 2.4%\n7. 012 4.6%'
    max_length = 24
    reply = bot.Messages.my_guess(y, preface='', max_length=max_length)
    assert reply == '\n2. 567 2.4%\n7. 012 4.6%'
    assert len(reply) <= max_length


def test_my_guess_honors_max_length_by_truncating_all():
    y = [
        deploy.Prediction(2, '567', 0.091),
        deploy.Prediction(4, '789', 0.012),
    ]
    # before truncation:
    # '\n2. 567 9.1%\n4. 789 1.2%'
    max_length = 20
    reply = bot.Messages.my_guess(y, preface='', max_length=max_length)
    assert reply == '\n2. 5 9.1%\n4. 7 1.2%'
    assert len(reply) <= max_length
