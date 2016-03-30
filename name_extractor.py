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
ANN_DETAILS_URL = 'http://cdn.animenewsnetwork.com/encyclopedia/api.xml'
ANN_ANIME_RATINGS_REPORT_ID = 172
TRAILING_KIND_RE = re.compile(r'\s+\([^)]+\)$')


def get_top_n_shows(n):
    """Returns top n shows from Anime News Network"""
    assert n <= 1000
    params = {'id': ANN_ANIME_RATINGS_REPORT_ID, 'nlist': n, 'nskip': 0}
    response = requests.get(ANN_REPORTS_URL, params=params)
    root = ElementTree.fromstring(response.content)
    return {
        'fields': ['id', 'name'],
        'items': map(lambda item: {'id': item.get('id'), 'name': _extract_item_name(item)}, root.iter('item'))
    }


def _extract_item_name(item):
    return TRAILING_KIND_RE.sub('', item.find('anime').text)


def list_characters(shows):
    ids = [show['id'] for show in shows['items']]
    params = {'anime': ids}
    response = requests.get(ANN_DETAILS_URL, params=params)
    root = ElementTree.fromstring(response.content)
    return {
        'fields': ['show_id', 'show_name', 'character_name'],
        'items': _extract_anime_characters(root),
    }


def _extract_anime_characters(root):
    for anime in root.iter('anime'):
        anime_id = anime.get('id')
        anime_name = anime.get('name')
        seen_names = set()
        for role in anime.findall("cast[@lang='JA']/role"):
            name = role.text
            if name not in seen_names:
                yield {
                    'anime_id': anime_id,
                    'anime_name': anime_name,
                    'name': role.text,
                }
                seen_names.add(name)


def print_csv(field_items, fileobj=sys.stdout, fields=None):
    writer = csv.writer(codecs.getwriter('utf8')(fileobj))
    fields = field_items['fields'] if fields is None else fields
    writer.writerow(fields)
    for row in field_items['items']:
        writer.writerow([row[field] for field in fields])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('what', choices=['shows', 'characters'])
    parser.add_argument('-n', type=int, default=50)
    args = parser.parse_args()

    if args.what == 'shows':
        print_csv(get_top_n_shows(args.n), fields=['name'])
    elif args.what == 'characters':
        shows = get_top_n_shows(args.n)
        print_csv(list_characters(shows), fields=['anime_name', 'name'])
