# -*- coding: utf-8 -*-
import collections
from functools import partial
from StringIO import StringIO
import json


def mock_get(content):
    def _mock_get(*args, **kwargs):
        return MockResponse(content)
    return _mock_get


class MockResponse(collections.namedtuple('Response', 'content')):
    def iter_content(self, chunk_size=1):
        return iter(partial(StringIO(self.content).read, chunk_size), '')

    def close(self):
        pass

    def json(self):
        return json.loads(self.content)
