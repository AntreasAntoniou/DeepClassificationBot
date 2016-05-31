# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math

import deploy
from deepanimebot.shortcuts import at_random


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
    def my_guess(cls, y, top_n=3, max_length=None, preface="Probable Anime:"):
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
        if max_length is not None and current_length > max_length:
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
        return reply[:max_length] if max_length is not None else reply


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
