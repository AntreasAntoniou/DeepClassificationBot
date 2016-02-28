# -*- coding: utf-8 -*-
import collections
import requests
import name_extractor


def test_top_n_shows(monkeypatch):
    for report, expected in [
            (two_items, ['Steins;Gate', 'Fullmetal Alchemist: Brotherhood']),
            (no_items, []),
    ]:
        monkeypatch.setattr(requests, 'get', mock_get(report))
        shows = name_extractor.get_top_n_shows(100)
        assert shows == expected


def mock_get(content):
    def _mock_get(*args, **kwargs):
        return MockResponse(content)
    return _mock_get


MockResponse = collections.namedtuple('Response', 'content')


two_items = '''
<report>
  <item id="11770">
    <anime href="/encyclopedia/anime.php?id=11770">Steins;Gate (TV)</anime>
  </item>
  <item id="10216">
    <anime href="/encyclopedia/anime.php?id=10216">Fullmetal Alchemist: Brotherhood (TV)</anime>
  </item>
</report>
'''

no_items = '''
<report>
</report>
'''
