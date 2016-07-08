# -*- coding: utf-8 -*-
'''
Command to initialize a workspace
'''
import click

from cmdbase import workspace_option


@click.command()
@workspace_option(required=False)
def init(workspace):
    print(workspace)


if __name__ == '__main__':
    init()
