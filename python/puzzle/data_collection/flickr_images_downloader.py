#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapted from:
    Phil Adams http://philadams.net

    Grab photos from Flickr for a set of keywords.  Considers only those photos
    with a CC non-commercial license, or more relaxed license (license ids 1,2,4,5
    at https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html)
"""
import random
import sys
import json
import os
from multiprocessing import Process

import times
import requests
import flickr_api
from PIL import Image
from resizeimage import resizeimage

API_KEY = 'd5c61f392ebcafccb59aee4c2019f16c'
API_SECRET = 'c1fe6c152d3f9600'
REST_ENDPOINT = 'https://api.flickr.com/services/rest/'
IMG_FNAME = './flickr_images/{query}/{id}.jpg'
IMG_DIR   = './flickr_images/{query}'
YMD = times.format(times.now(), 'Europe/London', fmt='%Y-%m-%d')
flickr_api.set_keys(api_key=API_KEY, api_secret=API_SECRET)


def log(msg):
    # Space 50 chars to the left
    sys.stdout.write('\r{0: <50}| '.format(msg))
    sys.stdout.flush()


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


def get_search_results(query, per_page, page=1):
    params = {
        'safe_search': '1',     # safest
        'media': 'photos',      # just photos
        'content_type': '1',    # just photos
        'privacy_filter': '1',  # public photos
        'license': '1,2,4,5',   # see README.md
        'per_page': per_page,   # max results per query
        'page': page,           # page of results
        'sort': 'relevance',
        'method': 'flickr.photos.search',
        'text': query
    }
    return do_request(params)


def save_image(urls, fname, size=528):
    if len(urls) == 0:
        print('Aborting... {}'.format(fname))
        return 0

    url = urls.pop(0)
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

    # Resize
    try:
        img = Image.open(fname)
        img = resizeimage.resize_cover(img, [size, size])
        img.save(fname, img.format)
        print('Downloaded & resized: {}'.format(fname))
        return 1
    except Exception as e:
        print('Retrying... {}: {}'.format(fname, e))
        try:
            os.remove(fname)
        except:
            pass
        return save_image(urls, fname, size)


def get_download_urls(photo, delta_ratio=0.2, min_pixel=528):
    try:
        sizes = get_photo_sizes(photo)['sizes']['size']
    except:
        print('get_download_url sizes missing')
        return None

    for size in sizes:
        if size['label'] == 'Original':

            # Ratio test
            w, h = int(size['width']), int(size['height'])
            ratio = h / w
            if abs(ratio - 1) > delta_ratio:
                return None

            # Min height/width
            return [s['source'] for s in sizes
                    if int(s['width']) >= min_pixel and int(s['height']) >= min_pixel]


def download_search(query, photos, download_target):
    img_dir = IMG_DIR.format(query=query)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    downloaded = 0
    for i, photo in enumerate(photos):
        log('checking photo %d/%d (%s)' % (i + 1, len(photos), query))

        img_fname = IMG_FNAME.format(query=query, **photo)

        if os.path.exists(img_fname):
            print('Exists: {}'.format(img_fname))
            downloaded += 1
            continue

        # Sizes
        urls = get_download_urls(photo)
        if urls is not None:
            downloaded += save_image(urls, img_fname)

        # Exit if has the right number of results
        if downloaded == download_target:
            return

    print('Photos missing: {}/{}'.format(downloaded, download_target))


def download_searches(search_results, download_target):
    for k, v in search_results.items():
        download_search(k, v, download_target)


def search(query, per_page, from_page):
    do_search = lambda p: get_search_results(query, 500, p)['photos']['photo']

    photos = []
    for page in range(from_page, from_page+int(per_page / 500)): # 500 is max allowed per_page
        try:
            photos += do_search(page)
        except:
            # Network error? Retrying...
            try:
                photos += do_search(page)
            except:
                # Skip...
                continue

    clean_query = query.replace(' ', '-')
    random.shuffle(photos)
    return clean_query, photos


def keywords_search(keywords, per_page, from_page):
    search_results = {}

    for i, keyword in enumerate(keywords):
        log('running keyword search... %d/%d (%s)' % (i + 1, len(keywords), keyword))
        _from_page = from_page
        if isinstance(from_page, dict):
            _from_page = from_page[keyword]
        print('keyword: {} : {}'.format(keyword, _from_page))
        k, v = search(keyword, per_page, _from_page)
        search_results[k] = v

    return search_results


def scrape_flickr(keywords, per_page, from_page=1):
    search_results = keywords_search(keywords, per_page, from_page)
    download_searches(search_results, per_page)


if __name__ == '__main__':
    keywords = [
        # 'animal',
        # 'art painting',
        # 'new york nyc skyline',
        'landscape mountains',
        # 'portrait',
        # 'space galaxy'
    ]
    from_page = {
        'animal': 35,
        'art painting': 35,
        'big cities': 35,
        'landscape': 35,
        'portrait': 35,
        'space galaxy': 18
    }

    jobs = []
    for kw in keywords:
        for i in range(1, 26):
            p = Process(target=scrape_flickr, args=([kw], 2000), kwargs=dict(from_page=i))
            p.start()
            jobs.append(p)

    for p in jobs:
        p.join()