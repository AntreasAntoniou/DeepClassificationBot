# -*- coding: utf-8 -*-
'''
Command-line interface.
'''
import click


if __name__ == '__main__':
    from cmdbase import cli, init_group
    from google_image_scraper import scrape
    from train import extract_data, run

    commands = click.CommandCollection(sources=[cli, init_group])
    commands()
