# -*- coding: utf-8 -*-
import requests
import name_extractor
import mocks


def test_top_n_shows(monkeypatch):
    for report, expected in [
            (two_shows, [
                {'id': '11770', 'name': 'Steins;Gate'},
                {'id': '10216', 'name': 'Fullmetal Alchemist: Brotherhood'}]),
            (no_shows, []),
    ]:
        monkeypatch.setattr(requests, 'get', mocks.mock_get(report))
        shows = name_extractor.get_top_n_shows(100)
        assert shows['items'] == expected


def test_list_characters(monkeypatch):
    for xml, expected in [
            (two_details, [
                {'anime_id': '11770', 'anime_name': 'Steins;Gate', 'name': 'Mayuri Shiina'},
                {'anime_id': '9701', 'anime_name': 'Clannad After Story', 'name': 'Fuuko Ibuki'},
            ]),
            (no_details, [])
    ]:
        monkeypatch.setattr(requests, 'get', mocks.mock_get(xml))
        shows = {
            'fields': ['show'],
            'items': [{'id': '11770'}, {'id': '10216'}],
        }
        characters = name_extractor.list_characters(shows)['items']
        assert list(characters) == expected


two_shows = '''
<report>
  <item id="11770">
    <anime href="/encyclopedia/anime.php?id=11770">Steins;Gate (TV)</anime>
  </item>
  <item id="10216">
    <anime href="/encyclopedia/anime.php?id=10216">Fullmetal Alchemist: Brotherhood (TV 2/2011)</anime>
  </item>
</report>
'''

no_shows = '''
<report>
</report>
'''

two_details = '''
<ann>
  <anime id="11770" name="Steins;Gate">
    <cast gid="3133619099" lang="DE">
      <role>Kurisu Makise</role>
      <person id="105651">Manuela Bäcker</person>
    </cast>
    <cast gid="3936561558" lang="JA">
      <role>Mayuri Shiina</role>
      <person id="53741">Kana Hanazawa</person>
    </cast>
  </anime>
  <anime id="10216" name="Fullmetal Alchemist: Brotherhood">
  </anime>
  <anime id="9701" gid="4003373163" type="TV" name="Clannad After Story">
    <cast gid="2029100002" lang="JA">
      <role>Fuuko Ibuki</role>
      <person id="13321">Ai Nonaka</person>
    </cast>
  </anime>
</ann>
'''

no_details = '''
<ann>
</ann>
'''
