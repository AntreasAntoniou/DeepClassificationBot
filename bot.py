# -*- coding: utf-8 -*-
'''
Twitter bot who replies with the best guesses of
what a @mention'ed image is.
'''
from __future__ import absolute_import
from __future__ import division

import time
import random
import functools
import logging
from multiprocessing import TimeoutError
import multiprocessing.pool

import requests
import numpy as np
import cv2
import h5py
import tweepy

import data
import deploy


INPUT_SHAPE = 128  # change it to your input image size
logging.basicConfig()
logger = logging.getLogger('bot')
logger.setLevel(logging.INFO)


# courtesy of http://stackoverflow.com/a/35139284/20226
def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(f):
        """Wrap the original function."""
        @functools.wraps(f)
        def func_wrapper(self, *args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(f, (self,) + args, kwargs)
            timeout = kwargs.pop('timeout_max_timeout', max_timeout) or max_timeout
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(timeout)
        return func_wrapper
    return timeout_decorator


def wait_like_a_human(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()

        rv = f(*args, **kwargs)
        if not rv:
            return
        api, action, args, kwargs = rv

        end = start + random.randint(1, 5)
        sleep = end - time.time()
        if sleep > 0:
            time.sleep(sleep)

        return getattr(api, action)(*args, **kwargs)
    return wrapper


class MockClassifier(object):
    def classify(self, *args, **kwargs):
        return at_random(
            "I hope a mock message like this won't get caught by Twitter's spam filter",
            "But I must explain to you how all this mistaken idea was born",
            "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis",
            "Excepteur sint occaecat cupidatat non proident",
        )


class ImageClassifier(object):
    def __init__(self, dataset_path):
        dataset = h5py.File(dataset_path, "r")
        self.average_image = dataset['mean'][:]

        n_categories = dataset['n_categories'].value
        self.model = deploy.load_model(input_shape=INPUT_SHAPE, n_outputs=n_categories)

        catname_to_categories = data.get_categories()
        self.category_to_catnames = {v: k for k, v in catname_to_categories.items()}

    def classify(self, cvimage):
        normalized = normalize_cvimage(cvimage, mean=self.average_image)
        return deploy.apply_model(normalized, self.model, self.category_to_catnames, multi=False)


def at_random(*messages):
    return random.choice(messages)


class Messages(object):
    @staticmethod
    def took_too_long():
        return at_random(
            "It took too long to get the image. Try again?",
        )

    @staticmethod
    def something_went_wrong():
        return at_random(
            "Something went wrong. Try again later?",
        )

    @staticmethod
    def not_an_image():
        return at_random(
            "That doesn't look like an image",
            "Are you sure it's an image?",
        )

    @staticmethod
    def unknown_image():
        return at_random(
            'I have no clue!',
            'Unknown',
        )

    @classmethod
    def my_guess(cls, y):
        if len(y):
            return y[0]
        else:
            return cls.unknown_image()


class StatusMessages(Messages):
    @staticmethod
    def give_me_an_image():
        return at_random(
            'Give me an image URL or attach it to your tweet',
            "I don't see an image. Tweet an image URL or attach it please",
        )


class DMMessages(Messages):
    @staticmethod
    def give_me_an_image():
        return at_random(
            'Give me an image URL',
            "I don't see an image. Message me an image URL please",
        )


class ReplyToTweet(tweepy.StreamListener):
    def __init__(self, screen_name, classifier, api=None):
        super(ReplyToTweet, self).__init__(api)
        self.screen_name = screen_name
        self.classifier = classifier

    @wait_like_a_human
    def on_direct_message(self, data):
        status = data.direct_message
        sender_name = status['sender']['screen_name']

        if sender_name == self.screen_name:
            return

        logger.debug(u"{0} incoming dm {1}".format(status['id'], status['text']))

        reply = self.get_reply(status['id'], status['entities'], DMMessages)
        return self.api, 'send_direct_message', tuple(), dict(user_id=status['sender']['id'], text=reply)

    @wait_like_a_human
    def on_status(self, status):
        sender_name = status.author.screen_name
        if sender_name == self.screen_name:
            return

        logger.debug(u"{0} incoming status {1}".format(status.id, status.text))

        if not status_mentions(status, self.screen_name):
            logger.debug("{0} doesn't mention {1}".format(status.id, self.screen_name))
            return

        reply = self.get_reply(status.id, status.entities, StatusMessages)
        status_text = '@{0} {1}'.format(sender_name, reply)
        return self.api, 'update_status', (status_text,), dict(in_reply_to_status_id=status.id)

    def get_reply(self, status_id, entities, messages):
        maybe_image_url = url_from_entities(entities)

        if not maybe_image_url:
            logger.debug("{0} doesn't have a URL".format(status_id))
            return messages.give_me_an_image()

        try:
            cvimage = fetch_cvimage_from_url(maybe_image_url)
        except TimeoutError:
            logger.debug("{0} timed out while fetching {1}".format(status_id, maybe_image_url))
            return messages.took_too_long()
        except Exception as e:
            logger.error("{0} error while fetching {1}: {2}".format(status_id, maybe_image_url, e))
            return messages.something_went_wrong()

        if cvimage is None:
            logger.debug("{0} no image found at {1}".format(status_id, maybe_image_url))
            return messages.not_an_image()

        y = self.classifier.classify(cvimage)
        reply = messages.my_guess(y)
        logger.debug("{0} reply: {1}".format(status_id, reply))
        return reply

    def on_error(self, status):
        if status == 420:
            # we are rate-limited.
            # returning False disconnects the stream
            return False


def status_mentions(status, screen_name):
    for mention in status.entities.get('user_mentions', []):
        if mention['screen_name'] == screen_name:
            return True
    return False


def url_from_entities(entities):
    for media in entities.get('media', []):
        if media['type'] == 'photo':
            return media['media_url']
    for url in entities.get('urls', []):
        return url['expanded_url']


@timeout(30)
def fetch_cvimage_from_url(url, maxsize=10 * 1024 * 1024):
    r = requests.get(url, timeout=5, stream=True)

    content = ''
    for chunk in r.iter_content(2048):
        content += chunk
        if len(content) > maxsize:
            r.close()
            raise ValueError('Response too large')

    cv2_img_flag = 0
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


# TODO: move to deploy.py (see get_data_from_file)
def normalize_cvimage(cvimage, size=INPUT_SHAPE, mean=None):
    result = data.resize(cvimage, size)
    if mean:
        result = result - mean
    return result / 255


def main(args):
    if args.debug:
        logger.setLevel(logging.DEBUG)

    auth = tweepy.OAuthHandler(args.consumer_key, args.consumer_secret)
    auth.set_access_token(args.access_token, args.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    screen_name = api.me().screen_name

    if args.mock:
        classifier = MockClassifier()
    else:
        classifier = ImageClassifier(args.dataset_path)

    stream = tweepy.Stream(auth=auth, listener=ReplyToTweet(screen_name, classifier, api))
    logger.info('Listening as {}'.format(screen_name))
    stream.userstream(track=[screen_name])


if __name__ == '__main__':
    import configargparse

    parser = configargparse.getArgumentParser()
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    parser.add('--consumer-key')
    parser.add('--consumer-secret')
    parser.add('--access-token')
    parser.add('--access-token-secret')
    parser.add('--dataset-path', default='data.hdf5')
    parser.add('--mock', action='store_true', default=False, help='test bot without model data')
    parser.add('--debug', action='store_true', default=False, help='set log level to debug')

    args = parser.parse_args()
    main(args)
