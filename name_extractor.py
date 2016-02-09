'''
Scrapes anime names from various sources.
'''
import re
import xml.etree.ElementTree as ElementTree
import requests


ANN_REPORTS_URL = 'http://www.animenewsnetwork.com/encyclopedia/reports.xml'
ANN_ANIME_RATINGS_REPORT_ID = 172
TRAILING_KIND_RE = re.compile(r'\s+\(\w+\)$')


def get_top_n_shows(n):
    '''Returns top n shows from Anime News Network'''
    assert n <= 1000
    params = {'id': ANN_ANIME_RATINGS_REPORT_ID, 'nlist': n, 'nskip': 0}
    response = requests.get(ANN_REPORTS_URL, params=params)
    root = ElementTree.fromstring(response.content)
    return map(_extract_item_name, root.iter('item'))


def _extract_item_name(item):
    return TRAILING_KIND_RE.sub('', item.find('anime').text)
