import glob
import os
import random
from enum import Enum

import numpy as np
import sys
import tqdm
from PIL import Image


class RelativePosition(Enum):
    RIGHT = "right"
    DOWN  = "down"


def extract_crop(img, crop_dim, crop_pos=(0, 0)):
    """
    Extracts a crop from a given image
    :param img: str
        image to crop
    :param crop_dim: (int, int)
        Dimensions (width, height) of the crop
    :param crop_pos: (int, int)
        Position of the top-left corner (x, y) of the crop
    :return: one crop of the image.
    """
    crop_width, crop_height = crop_dim
    crop_x, crop_y = crop_pos

    crop_img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    if crop_img.shape[:2] != crop_dim:
        raise ValueError("Image crop not square, has dims:{}, at location:{}".format(crop_img.shape[:2], crop_pos))

    return crop_img


def select_correct_crops_right(img, crop_dim=(48, 48)):
    """
      Extracts a set of correctly positioned crops to the right from a given image
      :param img
          Image to select crops from
      :param crop_dim: (int, int)
          Dimensions (width, height) of the crop
    """

    for x in range(0, 528, 48 * 2):
        for y in range(0, 528, 48):
            try:
                crop_left = extract_crop(img, crop_dim, (x, y))
                crop_right = extract_crop(img, crop_dim, (x + crop_dim[0], y))

                yield crop_left, crop_right, 0
            except ValueError:
                pass


def select_correct_crops_down(img, crop_dim=(48, 48)):
    """
      Extracts a set of correctly positioned crops down from a given image
     :param img
          Image to select crops from
      :param crop_dim: (int, int)
          Dimensions (width, height) of the crop
    """

    for x in range(0, 528, 48):
        for y in range(0, 528, 48 * 2):
            try:
                crop_up = extract_crop(img, crop_dim, (x, y))
                crop_down = extract_crop(img, crop_dim, (x, y + crop_dim[1]))

                yield crop_up, crop_down, 0
            except ValueError:
                pass


def select_incorrect_crops(img, crop_dim=(48, 48)):
    """
      Extracts a set of incorrectly positioned crops from a given image
     :param img
          Image to select crops from
      :param crop_dim: (int, int)
          Dimensions (width, height) of the crop
    """

    for x in range(0, 528, 48):  # why only 0-15?
        for y in range(0, 528, 48):
            try:
                # if the crop is not a square, pass and don't put in dataset
                crop = extract_crop(img, crop_dim, (x, y))
                crop_dim_x, crop_dim_y = crop_dim[0] * 5, crop_dim[1] * 5

                if (x, y) > (264, 264):
                    new_crop_pos = (x - crop_dim_x, y - crop_dim_y)
                elif x > 264 > y:
                    new_crop_pos = (x - crop_dim_x, y + crop_dim_y)
                elif x < 264 < y:
                    new_crop_pos = (x + crop_dim_x, y - crop_dim_y)
                else:
                    new_crop_pos = (x + crop_dim_x, y + crop_dim_y)

                crop2 = extract_crop(img, crop_dim, new_crop_pos)

                yield crop, crop2, 1
            except ValueError:
                pass


def create_training_set(images, rel_pos):
    """
    Creates 3 arrays containing image crops and whether they are adjacent or not
    :param data_path: path relative to the directory of this script
    :return: the 3 arrays
    """
    X1s = []
    X2s = []
    Ys = []

    select_correct_crops = {
        RelativePosition.RIGHT: select_correct_crops_right,
        RelativePosition.DOWN:  select_correct_crops_down,
    }

    sys.stdout.flush()
    for image_path in tqdm.tqdm(images, total=len(images)):
        img = np.array(Image.open(image_path))

        for X_1, X_2, Y in (select_correct_crops[rel_pos](img)):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)
        for X_1, X_2, Y in select_incorrect_crops(img):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)

    return X1s, X2s, Ys


def get_sets(image_class=None, test_set_portion=0.1, seed=42):
    """
    Returns the training & test sets for positions DOWN / RIGHT
    :param image_class: str or None
         image input class to use: e.g. animal, cities, or None for all
    :param test_set_portion: float
        portion of images to assign to test set, e.g. 0.1
    :param seed: int
        seed number for random generator (same seed produces same sequence of random numbers).
    :return: [train_down, train_right, test_down, test_right]
        Four sets of the form (X1s, X2s, Ys)
    """
    # Generate training / test sets
    test_images, training_images = get_image_sets(image_class, test_set_portion, seed)

    args = [
        (training_images, RelativePosition.DOWN),
        (training_images, RelativePosition.RIGHT),
        (test_images,     RelativePosition.DOWN),
        (test_images,     RelativePosition.RIGHT),
    ]

    res = []
    for arg in args:
        res.append(create_training_set(*arg))
    return res


def get_image_sets(image_class=None, test_set_portion=0.1, seed=42):
    # Find images of interest
    curr_dir = os.path.dirname(__file__)
    img_dir = os.path.realpath(os.path.join(curr_dir, '../../../images'))
    image_pattern = os.path.join(img_dir, '**/*.jpg')

    if image_class is not None:
        image_pattern = os.path.join(img_dir, image_class, '*.jpg')
    print('Searching for images at: {}'.format(image_pattern))

    # Generate training / test images
    if seed is not None:
        # Seed to generate the same sequence of random.random() numbers
        random.seed(seed)

    training_images = []
    test_images = []
    for img in glob.glob(image_pattern, recursive=True):
        if random.random() <= test_set_portion:
            test_images.append(img)
        else:
            training_images.append(img)

    return test_images, training_images


if __name__ == "__main__":
    [train_down, train_right, test_down, test_right] = get_sets(image_class='animals')
    print('Training sets size: d={}, r={}'.format(len(train_down), len(train_right)))
    X1s, X2s, Ys = train_down
    print('X1s={}, X2s={}, Ys={}'.format(len(X1s), len(X2s), len(Ys)))
