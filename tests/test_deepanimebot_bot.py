# -*- coding: utf-8 -*-

import deploy
from deepanimebot import bot


def test_my_guess_honors_max_length_by_truncating_longest():
    y = [
        deploy.Prediction(2, '567890', 0.024),
        deploy.Prediction(7, '012', 0.046),
    ]
    # before truncation:
    # '\n2. 567890 2.40%\n7. 012 4.60%'
    max_length = 26
    reply = bot.Messages.my_guess(y, preface='', max_length=max_length)
    assert reply == '\n2. 567 2.40%\n7. 012 4.60%'
    assert len(reply) <= max_length


def test_my_guess_honors_max_length_by_truncating_all():
    y = [
        deploy.Prediction(2, '567', 0.091),
        deploy.Prediction(4, '789', 0.012),
    ]
    # before truncation:
    # '\n2. 567 9.10%\n4. 789 1.20%'
    max_length = 22
    reply = bot.Messages.my_guess(y, preface='', max_length=max_length)
    assert reply == '\n2. 5 9.10%\n4. 7 1.20%'
    assert len(reply) <= max_length
