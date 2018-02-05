import os
import cv2

from puzzle.scripts.resize_all_images import resize_all
from puzzle.tools.utils import input_image, img_read, input_directory

rel_path         = '../../../images'
rel_path_resized = '../../../images/resized'
rel_path_merged  = '../../../images/merged'
rel_path_merged_test  = '../../../images/merge_test'

def mkpath(p):
    curr_dir = os.path.dirname(__file__)
    return os.path.realpath(os.path.join(curr_dir, p))

def merge_loulou(dir_in, img1, img2):

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


def merge(dir_in, img1, img2):

    # Create counter to renamed merged files
    dir_out = os.path.realpath(os.path.join(dir_in, rel_path_merged_test))
    n = len(os.listdir(dir_out))

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
    m = n + 1
    dir_out = mkpath(rel_path_merged)
    os.makedirs(dir_out, exist_ok=True)
    crop_name = os.path.join(dir_out, str(m)+".jpg")
    cv2.imwrite(crop_name, img2)


def merge_all(dir_in):

    for img in os.listdir(dir_in):
        path1 = os.path.join(dir_in, img)
        img = img_read(path1)
        for img2 in os.listdir(dir_in):
            path2 = os.path.join(dir_in, img2)
            img2 = img_read(path2)
            if path1 != path2:
                merge(dir_in, img, img2)

if __name__ == "__main__":
    dir_in = input_directory()
    resized_dir = os.path.join(dir_in, "resized")
    merge_all(resized_dir)