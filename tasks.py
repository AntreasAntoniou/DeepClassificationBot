# -*- coding: utf-8 -*-
'''
Bot deployment tasks
'''
from __future__ import absolute_import

import os
import subprocess
import urllib

from configargparse import ConfigFileParser
from temporary import temp_dir
import click


DEFAULT_INSTANCE_NAME = 'bot'
DEFAULT_ZONE = 'us-central1-a'
DEFAULT_MACHINE_TYPE = 'f1-micro'
LOGGING_AGENT_INSTALL_SCRIPT = 'https://dl.google.com/cloudagents/install-logging-agent.sh'


@click.group()
def bot():
    pass


@bot.command()
@click.option('--name', default=DEFAULT_INSTANCE_NAME)
@click.option('--zone', default=DEFAULT_ZONE)
@click.option('--machine-type', default=DEFAULT_MACHINE_TYPE)
@click.option('--address', default=None)
@click.option('--bot-config', default='bot.ini')
@click.option('--stackdriver-logging/--no-stackdriver-logging', default=False, help='Install logging agent and add config to collect logs')
@click.pass_context
def create_instance(ctx, name, zone, machine_type, address, bot_config, stackdriver_logging):
    args = [
        'gcloud', 'compute', 'instances', 'create', name,
        '--image', 'container-vm',
        '--zone', zone,
        '--machine-type', machine_type,
    ]

    if address:
        args.append('--address')
        args.append(address)

    with open(bot_config) as f:
        bot_config_content = ConfigFileParser().parse(f)
    if len(bot_config_content):
        secret_args = [
            '--metadata',
            ','.join('='.join(item) for item in bot_config_content.items()),
        ]

    with temp_dir() as d:
        # add metadata from file
        args.append('--metadata-from-file')
        metadata_files = ['google-container-manifest=etc/containers.yaml']
        startup_script_path = os.path.join(d, 'startup-script.sh')
        if stackdriver_logging:
            urllib.urlretrieve(LOGGING_AGENT_INSTALL_SCRIPT, startup_script_path)
        with open(startup_script_path, 'a') as f:
            f.write('\nmkdir -p /var/log/bot\n')
        metadata_files.append('startup-script={}'.format(startup_script_path))
        args.append(','.join(metadata_files))

        confirm = "Create the following instance? (+{num} metadata from {config})\n{command}".format(
            num=len(bot_config_content),
            config=bot_config,
            command=' '.join(args))
        click.confirm(confirm, abort=True)
        subprocess.call(args + secret_args)
        if stackdriver_logging:
            ctx.invoke(copy_fluentd_conf, name=name, zone=zone)


@bot.command()
@click.option('--name', default=DEFAULT_INSTANCE_NAME)
@click.option('--zone', default=DEFAULT_ZONE)
def copy_fluentd_conf(name, zone):
    args = [
        'gcloud', 'compute', 'copy-files',
        'etc/fluentd.conf', '{}:~/bot.conf'.format(name),
        '--zone', zone,
    ]
    print(' '.join(args))
    subprocess.call(args)
    args = [
        'gcloud', 'compute', 'ssh',
        name,
        '--zone', zone,
        '--command', 'sudo mkdir -p /etc/google-fluentd/config.d && sudo mv bot.conf /etc/google-fluentd/config.d/ && sudo service google-fluentd restart || true'
    ]
    print(' '.join(args))
    subprocess.call(args)


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
