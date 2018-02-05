import os
import cv2

from puzzle.scripts.resize_all_images import resize_all
from puzzle.tools.utils import input_image, img_read


def mkpath(p):
    curr_dir = os.path.dirname(__file__)
    return os.path.realpath(os.path.join(curr_dir, p))

rel_path         = '../../../images'
rel_path_resized = '../../../images/resized'
rel_path_merged  = '../../../images/merged'

#resize_all(mkpath(rel_path), size=200)

img_path_1 = input_image(images_relpath=rel_path)
img_path_2 = input_image(images_relpath=rel_path)

# Load image
img1 = img_read(img_path_1)
img2 = img_read(img_path_2)

height, width = img1.shape[:2]
height2, width2 = img1.shape[:2]
assert (height == height2)
assert (width == width2)

half_width = width // 2

# Iterate in the range(begin, end)
for y in range(0, height):
    img2[y, half_width:width] = img2[y, 0:half_width]
    img2[y, 0:half_width]     = img1[y, half_width:width]

# Save
dir_out = mkpath(rel_path_merged)
os.makedirs(dir_out, exist_ok=True)
crop_name = os.path.join(dir_out, 'merged_image.jpg')
cv2.imwrite(crop_name, img2)

## dir_in = input_directory()
#    img_path_1 = input_image(images_relpath=dir_in)
#    img_path_2 = input_image(images_relpath=dir_in)
    # Load image
#    img1 = img_read(img_path_1)
#   img2 = img_read(img_path_2)
#   merge(dir_in, img1, img2)#

def merge(dir_in, img1, img2):

    img_path_1 = input_image(images_relpath=rel_path)
    img_path_2 = input_image(images_relpath=rel_path)

    # Load image
    img1 = img_read(img_path_1)
    img2 = img_read(img_path_2)

    height, width = img1.shape[:2]
    height2, width2 = img1.shape[:2]
    assert (height == height2)
    assert (width == width2)

    half_width = width // 2

    # Iterate in the range(begin, end)
    for y in range(0, height):
        img2[y, half_width:width] = img2[y, 0:half_width]
        img2[y, 0:half_width]     = img1[y, half_width:width]

    # Save
    img_name_1, ext = os.path.splitext(os.path.basename(img_path_1))
    img_name_2, _ = os.path.splitext(os.path.basename(img_path_2))

    dir_out = mkpath(rel_path_merged)
    os.makedirs(dir_out, exist_ok=True)
    crop_name = os.path.join(dir_out, "merge-{}-{}.{}".format(img_name_1, img_name_2, ext))
    cv2.imwrite(crop_name, img2)