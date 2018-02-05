#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
from resizeimage import resizeimage


def img_resize(img_path_in, img_path_out, size=512):
    with open(img_path_in, 'rb') as fd_img:
        img = Image.open(fd_img)
        img = resizeimage.resize_cover(img, [size, size])
        img = img.convert("RGB")
        img.save(img_path_out, img.format)