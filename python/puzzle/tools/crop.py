import os
import tempfile

import cv2

from puzzle.tools.utils import img_read


def crop_one(img_path, crop_dim, crop_pos=(0, 0)):
    """
    Extracts a crop from a given image
    :param img_path: str
        Path to the image to crop from
    :param crop_pos: (int, int)
        Position of the top-left corner (x, y) of the crop
    :param crop_dim: (int, int)
        Dimensions (height, width) of the crop
    :return: str, path of the directory containing the cropped images.
    """
    crop_height, crop_width = crop_dim
    crop_x, crop_y = crop_pos

    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()

    # Load image
    img = img_read(img_path)

    crop_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    _, img_extension = os.path.splitext(img_path)
    crop_name = os.path.join(tmp_dir, 'crop{}'.format(img_extension))
    cv2.imwrite(crop_name, crop_img)
    
    return crop_name


def crop_all(img_path, block_dim):
    """
    Crops an image into blocks of given width / height
    :param img_path: str
        Path of the image to crop from
    :param block_dim: (int, int)
        Tuple of block_width, block_height
    :return: str, path of the directory containing the cropped images.
    """
    block_height, block_width = block_dim

    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()

    # Load image
    img = img_read(img_path)
    height, width = img.shape[:2]

    # Iterate in the range(begin, end, step)
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            crop_name = os.path.join(tmp_dir, 'img_y-%d_x-%d.png' % (y, x))
            crop_img  = img[y:y+block_height, x:x+block_width]
            cv2.imwrite(crop_name, crop_img)

    return tmp_dir