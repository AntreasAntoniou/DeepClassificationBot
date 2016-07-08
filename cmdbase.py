# -*- coding: utf-8 -*-
'''
Click groups for a command-line interface.
'''
from collections import namedtuple
import functools

import click
from click_default_group import DefaultGroup


Workspace = namedtuple('Workspace', 'home')
pass_workspace = click.make_pass_decorator(Workspace)


def workspace_option(required=False):
    if required:
        option_type = click.Path(exists=True, file_okay=False)
    else:
        option_type = click.Path()

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return click.option(
            '--workspace',
            envvar='WORKSPACE',
            default='workspace',
            type=option_type)(wrapper)

    return decorator


@click.group(cls=DefaultGroup)
@workspace_option(required=True)
@click.pass_context
def cli(ctx, workspace):
    ctx.obj = Workspace(workspace)


@click.group()
def init_group():
    pass


@init_group.command()
@workspace_option(required=False)
def init(workspace):
    print(workspace)
