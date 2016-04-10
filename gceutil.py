# -*- coding: utf-8 -*-
'''
Utils for working in a GCE environment.

`detect_gce_environment` courtesy of:
https://github.com/google/oauth2client/blob/8f4d9164d98b23f3cac0f0785312a50ef22899e4/oauth2client/client.py
'''
import missing `from __future__ import absolute_import

import socket

import six
import requests


_GCE_METADATA_HOST = '169.254.169.254'
_METADATA_FLAVOR_HEADER = 'Metadata-Flavor'
_DESIRED_METADATA_FLAVOR = 'Google'
_METADATA_HEADERS = {_METADATA_FLAVOR_HEADER: _DESIRED_METADATA_FLAVOR}


def detect_gce_environment(logger):
    """Determine if the current environment is Compute Engine.
    Returns:
        Boolean indicating whether or not the current environment is Google
        Compute Engine.
    """
    # NOTE: The explicit ``timeout`` is a workaround. The underlying
    #       issue is that resolving an unknown host on some networks will take
    #       20-30 seconds; making this timeout short fixes the issue, but
    #       could lead to false negatives in the event that we are on GCE, but
    #       the metadata resolution was particularly slow. The latter case is
    #       "unlikely".
    connection = six.moves.http_client.HTTPConnection(
        _GCE_METADATA_HOST, timeout=1)

    try:
        connection.request('GET', '/', headers=_METADATA_HEADERS)
        response = connection.getresponse()
        if response.status == six.moves.http_client.OK:
            return (response.getheader(_METADATA_FLAVOR_HEADER) ==
                    _DESIRED_METADATA_FLAVOR)
    except socket.error:  # socket.timeout or socket.error(64, 'Host is down')
        logger.info('Timeout attempting to reach GCE metadata service.')
        return False
    finally:
        connection.close()


def get_metadata(attrs):
    return {attr: requests.get(metadata_url(attr), headers=_METADATA_HEADERS).content for attr in attrs}


def metadata_url(attr):
    return "http://{0}/computeMetadata/v1/instance/attributes/{1}".format(_GCE_METADATA_HOST, attr)
