#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapted from:
    Phil Adams http://philadams.net

    Grab photos from Flickr for a set of keywords.  Considers only those photos
    with a CC non-commercial license, or more relaxed license (license ids 1,2,4,5
    at https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html)
"""

import sys
import time
import json
import os
import glob
from pprint import pprint
import argparse

import times
import requests
import flickr_api

API_KEY = 'd5c61f392ebcafccb59aee4c2019f16c'
API_SECRET = 'c1fe6c152d3f9600'
REST_ENDPOINT = 'https://api.flickr.com/services/rest/'
IMG_URL   = 'http://farm{farm}.staticflickr.com/{server}/{id}_{secret}_z.jpg'
IMG_FNAME = './flickr_images/{query}/{id}.jpg'
IMG_DIR   = './flickr_images/{query}'
YMD = times.format(times.now(), 'Europe/London', fmt='%Y-%m-%d')
flickr_api.set_keys(api_key=API_KEY, api_secret=API_SECRET)


def do_request(extra_params):
    default_params = {
        'api_key': API_KEY,
        'format': 'json',
        'nojsoncallback': '1',  # no jsonp, only json
    }
    params = dict(default_params, **extra_params)
    response = requests.get(REST_ENDPOINT, params=params)
    return json.loads(response.text)


def get_photo_info(photo):
    params = {
        'method': 'flickr.photos.getInfo',
        'photo_id': photo['id'],
        'secret': photo['secret'],
    }
    return do_request(params)


def get_photo_sizes(photo):
    params = {
        'method': 'flickr.photos.getSizes',
        'photo_id': photo['id'],
        'secret': photo['secret'],
    }
    return do_request(params)


def get_search_results(query, per_page):
    params = {
        'safe_search': '1',  # safest
        'media': 'photos',  # just photos
        'content_type': '1',  # just photos
        'privacy_filter': '1',  # public photos
        'license': '1,2,4,5',  # see README.md
        'per_page': str(per_page),  # max results per query
        'sort': 'relevance',
        'method': 'flickr.photos.search',
        'text': query
    }
    return do_request(params)


def save_image(url, fname):
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
        return True


def download_search(query, results):
    meta = dict(query=query)
    img_dir = IMG_DIR.format(**meta)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    for i, photo in enumerate(results['photos']['photo']):
        sys.stdout.write('\rdownloading photo %d/%d (%s)           ' %
                         (i + 1,
                          len(results['photos']['photo']),
                          meta['query']))
        sys.stdout.flush()

        # Sizes
        sizes = get_photo_sizes(photo)['sizes']['size']
        for size in sizes:
            if size['label'] == 'Original':
                print('Original dims {width}x{height}: {url}'.format(**size))
                ratio = int(size['height']) / int(size['width'])
                if abs(ratio - 1) < 0.2:
                    print('Valid ratio: ', ratio)

        img_url   = IMG_URL.format(**photo)
        img_fname = IMG_FNAME.format(**dict(**meta, **photo))
        save_image(img_url, img_fname)


def download_searches(search_results):
    for k, v in search_results.items():
        download_search(k, v)


def search(query, per_page=5):
    data = get_search_results(query, per_page)
    clean_query = query.replace(' ', '-')
    return clean_query, data


def keywords_search(keywords):
    search_results = {}
    for i, keyword in enumerate(keywords):
        sys.stdout.write('\rrunning keyword search... %d/%d (%s)       ' %
                         (i + 1, len(keywords), keyword))
        sys.stdout.flush()
        k, v = search(keyword)
        search_results[k] = v
    return search_results


def scape_flickr(keywords=None):
    if keywords is None:
        keywords = ['animal', 'art', 'cities', 'landscape', 'portrait', 'space']

    search_results = keywords_search(keywords)
    download_searches(search_results)


if __name__ == '__main__':
    scape_flickr()

