# -*- coding: utf-8 -*-
'''
Twitter bot who replies with the best guesses of
what a @mention'ed image is.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
import logging
import os
import random
import time

import tweepy

import deploy
import gceutil
from deepanimebot import classifiers
from deepanimebot import exceptions as exc
from deepanimebot import messages


INPUT_SHAPE = 128  # change it to your input image size
TWEET_MAX_LENGTH = 140
logging.basicConfig()
logger = logging.getLogger('bot')
logger.setLevel(logging.INFO)


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

        reply = self.get_reply(status['id'], status['entities'], TWEET_MAX_LENGTH - len('d {} '.format(sender_name)), messages.DMMessages)
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
        reply = self.get_reply(status.id, status.entities, TWEET_MAX_LENGTH - len(prefix), messages.StatusMessages)
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
            y = self.classifier.classify(url=maybe_image_url)
        except exc.TimeoutError:
            logger.debug("{0} timed out while classifying {1}".format(status_id, maybe_image_url))
            return messages.took_too_long()
        except exc.NotImage:
            logger.debug("{0} no image found at {1}".format(status_id, maybe_image_url))
            return messages.not_an_image()
        except exc.RemoteError as e:
            logger.debug("{0} remote error {1}".format(status_id, e))
            return e.message
        except Exception as e:
            logger.error("{0} error while classifying {1}: {2}".format(status_id, maybe_image_url, e))
            return messages.something_went_wrong()

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


def main(args):
    if args.debug:
        logger.setLevel(logging.DEBUG)

    auth = tweepy.OAuthHandler(args.consumer_key, args.consumer_secret)
    auth.set_access_token(args.access_token, args.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    screen_name = api.me().screen_name

    if args.classifier == 'mock':
        classifier = classifiers.MockClassifier()
    elif args.classifier == 'local':
        classifier = classifiers.URLClassifier(classifiers.ImageClassifier(args.dataset_path, INPUT_SHAPE))
    elif args.classifier == 'remote':
        classifier = classifiers.RemoteClassifier(args.remote_endpoint)

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
    parser.add('--classifier', choices=['mock', 'local', 'remote'], default='mock', help='Which classifier to use')
    parser.add('--dataset-path', default='data/data.hdf5', help='Path to dataset when using a local calssifier')
    parser.add('--remote-endpoint', default=None, help='API endpoint to call when using a remote classifier')
    parser.add('--silent', action='store_true', default=False, help='Run bot without actually replying')
    parser.add('--debug', action='store_true', default=False, help='Set log level to debug')

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if gceutil.detect_gce_environment(logger):
            attrname_env_varnames = {action.dest.replace('_', '-'): action.env_var
                                     for action in parser._actions if action.env_var}
            metadata = gceutil.get_metadata(attrname_env_varnames.keys())
            environ = dict(os.environ)
            environ.update({attrname_env_varnames[attr]: value for attr, value in metadata.items()})
            args = parser.parse_args(env_vars=environ)
        else:
            raise

    main(args)
