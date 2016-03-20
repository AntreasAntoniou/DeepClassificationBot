'''
Scrapes anime names from various sources.
'''
import sys
import codecs
import re
from backports import csv
import xml.etree.ElementTree as ElementTree
import requests


ANN_REPORTS_URL = 'http://www.animenewsnetwork.com/encyclopedia/reports.xml'
ANN_ANIME_RATINGS_REPORT_ID = 172
TRAILING_KIND_RE = re.compile(r'\s+\([^)]+\)$')


def get_top_n_shows(n):
    """Returns top n shows from Anime News Network"""
    assert n <= 1000
    params = {'id': ANN_ANIME_RATINGS_REPORT_ID, 'nlist': n, 'nskip': 0}
    response = requests.get(ANN_REPORTS_URL, params=params)
    root = ElementTree.fromstring(response.content)
    return {
        'fields': ['show'],
        'items': map(lambda item: [_extract_item_name(item)], root.iter('item'))
    }


def _extract_item_name(item):
    return TRAILING_KIND_RE.sub('', item.find('anime').text)


def list_characters(shows):
    pass


def print_csv(field_items):
    writer = csv.writer(codecs.getwriter('utf8')(sys.stdout))
    writer.writerow(field_items['fields'])
    for row in field_items['items']:
        writer.writerow(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('what', choices=['shows', 'characters'])
    parser.add_argument('-n', type=int, default=50)
    args = parser.parse_args()

    if args.what == 'shows':
        print_csv(get_top_n_shows(args.n))
    elif args.what == 'characters':
        shows = get_top_n_shows(args.n)
        print_csv(list_characters(shows))
