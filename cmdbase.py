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
        return click.option(
            '--workspace',
            envvar='WORKSPACE',
            default='workspace',
            type=option_type)(fn)

    return decorator


@click.group()
@workspace_option(required=True)
@click.pass_context
def cli(ctx, workspace):
    ctx.obj = Workspace(workspace)
