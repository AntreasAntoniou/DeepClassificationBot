# -*- coding: utf-8 -*-
'''
Twitter bot who replies with the best guesses of
what a @mention'ed image is.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import random
import math
import functools
import logging
from multiprocessing import TimeoutError
import multiprocessing.pool

import cv2
import h5py
import requests
import tweepy

import data
import deploy
import gceutil
import numpy as np

INPUT_SHAPE = 128  # change it to your input image size
TWEET_MAX_LENGTH = 140
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
        catname_to_categories = data.get_categories()
        self.category_to_catnames = {v: k for k, v in catname_to_categories.items()}
        self.model = deploy.load_model(input_shape=INPUT_SHAPE, n_outputs=len(catname_to_categories))

        dataset = h5py.File(dataset_path, "r")
        self.average_image = dataset['mean'][:]

    def classify(self, cvimage):
        normalized = normalize_cvimage(cvimage, mean=self.average_image)
        return deploy.apply_model(normalized, self.model, self.category_to_catnames)


def at_random(*messages):
    return random.choice(messages)


class Messages(object):
    '''Each method is expected to return a message of length under TWEET_MAX_LENGTH.
    '''
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
    def my_guess(cls, y, top_n=3, max_length=TWEET_MAX_LENGTH, preface="Probable Anime:"):
        if not len(y):
            return cls.unknown_image()

        pred_lines = []
        max_category_length = 0
        max_category_length_index = 0

        for i, pred in enumerate(y[:top_n]):
            pred_lines.append(deploy.Prediction(
                "{}.".format(pred.rank),
                pred.category,
                "{:.2%}".format(pred.probability),
            ))
            if max_category_length < len(pred.category):
                max_category_length_index = i
                max_category_length = len(pred.category)

        newline_count = len(pred_lines)
        pred_length = sum(sum(map(len, pred)) + len(pred) - 1 for pred in pred_lines)
        current_length = len(preface) + newline_count + pred_length

        # truncate category name(s) if needed
        if current_length > max_length:
            lengthy_pred = pred_lines[max_category_length_index]
            excess_length = current_length - max_length
            # don't penalize the longest category if it's going to be truncated too much
            if len(lengthy_pred.category) * 0.5 < excess_length:
                subtract_from_everyone_length = int(math.ceil(excess_length / len(pred_lines)))
                pred_lines = [
                    deploy.Prediction(
                        pred.rank, pred.category[:-subtract_from_everyone_length], pred.probability)
                    for pred in pred_lines]
            else:
                shortened_pred = deploy.Prediction(
                    lengthy_pred.rank, lengthy_pred.category[:-excess_length], lengthy_pred.probability)
                pred_lines[max_category_length_index] = shortened_pred

        reply = "{}\n{}".format(preface, "\n".join(" ".join(pred) for pred in pred_lines))
        return reply[:max_length]


class StatusMessages(Messages):
    @staticmethod
    def give_me_an_image():
        return at_random(
            'Give me a direct image URL or attach it to your tweet',
            "I don't see an image. Tweet a direct image URL or attach it please",
        )


class DMMessages(Messages):
    @staticmethod
    def give_me_an_image():
        return at_random(
            'Give me a direct image URL',
            "I don't see an image. Message me a direct image URL please",
        )


class ReplyToTweet(tweepy.StreamListener):
    def __init__(self, screen_name, classifier, api=None, silent=False):
        super(ReplyToTweet, self).__init__(api)
        self.screen_name = screen_name
        self.classifier = classifier
        self.silent = silent

    @wait_like_a_human
    def on_direct_message(self, data):
        status = data.direct_message
        sender_name = status['sender']['screen_name']

        if sender_name == self.screen_name:
            return

        logger.debug(u"{0} incoming dm {1}".format(status['id'], status['text']))

        reply = self.get_reply(status['id'], status['entities'], TWEET_MAX_LENGTH - len('d {} '.format(sender_name)), DMMessages)
        if self.silent:
            return
        return self.api, 'send_direct_message', tuple(), dict(user_id=status['sender']['id'], text=reply)

    @wait_like_a_human
    def on_status(self, status):
        sender_name = status.author.screen_name
        if sender_name == self.screen_name:
            return

        logger.debug(u"{0} incoming status {1}".format(status.id, status.text))

        if retweets_me(status, self.screen_name):
            logger.debug("{0} is a retweet".format(status.id))
            return

        if not status_mentions(status, self.screen_name):
            logger.debug("{0} doesn't mention {1}".format(status.id, self.screen_name))
            return

        prefix = '@{0} '.format(sender_name)
        reply = self.get_reply(status.id, status.entities, TWEET_MAX_LENGTH - len(prefix), StatusMessages)
        status_text = prefix + reply
        if self.silent:
            return
        return self.api, 'update_status', (status_text,), dict(in_reply_to_status_id=status.id)

    def get_reply(self, status_id, entities, max_length, messages):
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
        reply = messages.my_guess(y, max_length)
        logger.debug("{0} reply: {1}".format(status_id, reply))
        return reply

    def on_error(self, status):
        if status == 420:
            # we are rate-limited.
            # returning False disconnects the stream
            return False


def retweets_me(status, screen_name):
    retweeted_status = getattr(status, 'retweeted_status', None)
    if retweeted_status is None:
        return False
    return retweeted_status.author.screen_name == screen_name


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
    req = requests.get(url, timeout=5, stream=True)
    content = ''
    for chunk in req.iter_content(2048):
        content += chunk
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    cv2_img_flag = cv2.CV_LOAD_IMAGE_COLOR
    image = cv2.imdecode(img_array, cv2_img_flag)
    return image


# TODO: move to deploy.py (see get_data_from_file)
def normalize_cvimage(cvimage, size=INPUT_SHAPE, mean=None):
    result = data.resize(cvimage, size)
    if mean is not None:
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

    stream = tweepy.Stream(auth=auth, listener=ReplyToTweet(screen_name, classifier, api, args.silent))
    logger.info('Listening as {}'.format(screen_name))
    stream.userstream(track=[screen_name])


if __name__ == '__main__':
    import configargparse

    parser = configargparse.getArgumentParser()

    parser.add('-c', '--config', required=False, is_config_file=True, help='Config file path. See bot.ini.example')
    parser.add('--consumer-key', required=True, env_var='CONSUMER_KEY', help='Twitter app consumer key')
    parser.add('--consumer-secret', required=True, env_var='CONSUMER_SECRET', help='Twitter app consumer secret')
    parser.add('--access-token', required=True, env_var='ACCESS_TOKEN', help='Twitter access token')
    parser.add('--access-token-secret', required=True, env_var='ACCESS_TOKEN_SECRET', help='Twitter access token secret')
    parser.add('--dataset-path', default='data/data.hdf5')
    parser.add('--mock', action='store_true', default=False, help='Test bot without model data')
    parser.add('--silent', action='store_true', default=False, help='Test bot without actually replying')
    parser.add('--debug', action='store_true', default=False, help='Set log level to debug')

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if gceutil.detect_gce_environment(logger):
            attr_env_vars = {action.dest.replace('_', '-'): action.env_var
                             for action in parser._actions if action.env_var}
            metadata = gceutil.get_metadata(attr_env_vars.keys())
            env_vars = {attr_env_vars[attr]: value for attr, value in metadata.items()}
            args = parser.parse_args(env_vars=env_vars)
        else:
            raise

    main(args)
