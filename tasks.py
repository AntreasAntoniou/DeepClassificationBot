# -*- coding: utf-8 -*-
'''
Bot deployment tasks
'''
from __future__ import absolute_import

import subprocess

from configargparse import ConfigFileParser
import click


DEFAULT_INSTANCE_NAME = 'bot'
DEFAULT_ZONE = 'us-central1-a'


@click.group()
def bot():
    pass


@bot.command()
@click.option('--name', default=DEFAULT_INSTANCE_NAME)
@click.option('--zone', default=DEFAULT_ZONE)
@click.option('--config', default='bot.ini')
def create_instance(name, zone, config):
    args = [
        'gcloud', 'compute', 'instances', 'create', name,
        '--image', 'container-vm',
        '--zone', zone,
        '--machine-type', 'f1-micro',
        '--metadata-from-file', 'google-container-manifest=etc/containers.yaml',
    ]
    with open(config) as f:
        bot_config = ConfigFileParser().parse(f)
    if len(bot_config):
        secret_args = [
            '--metadata',
            ','.join('='.join(item) for item in bot_config.items()),
        ]

    confirm = "Create the following instance? (+{num} metadata from {config})\n{command}".format(
        num=len(bot_config),
        config=config,
        command=' '.join(args))
    click.confirm(confirm, abort=True)
    subprocess.call(args + secret_args)


@bot.command()
@click.option('--name', default=DEFAULT_INSTANCE_NAME)
@click.option('--zone', default=DEFAULT_ZONE)
def delete_instance(name, zone):
    args = [
        'gcloud', 'compute', 'instances', 'delete', name,
        '--zone', zone,
    ]
    subprocess.call(args)


if __name__ == '__main__':
    bot()
