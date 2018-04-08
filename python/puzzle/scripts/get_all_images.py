import glob
import os
import pickle
from enum import Enum
from multiprocessing import Process

import numpy as np
import tqdm
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from resizeimage import resizeimage


class RelativePosition(Enum):
    RIGHT = "right"
    DOWN = "down"


def crop_one(img, crop_dim, crop_pos=(0, 0)):
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

    if isinstance(img, str):
        img = Image.open(img)
        img = resizeimage.resize_cover(img, [528, 528])
        img = np.array(img)

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
                crop_left = crop_one(img, crop_dim, (x, y))
                crop_right = crop_one(img, crop_dim, (x + crop_dim[0], y))

                yield crop_left, crop_right, 0
            except ValueError:
                pass


# TODO: perhaps merge the 2 functions together?

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
                crop_up = crop_one(img, crop_dim, (x, y))
                crop_down = crop_one(img, crop_dim, (x, y + crop_dim[1]))

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
                crop = crop_one(img, crop_dim, (x, y))
                crop_dim_x, crop_dim_y = crop_dim[0] * 5, crop_dim[1] * 5

                if (x, y) > (264, 264):
                    new_crop_pos = (x - crop_dim_x, y - crop_dim_y)
                elif x > 264 > y:
                    new_crop_pos = (x - crop_dim_x, y + crop_dim_y)
                elif x < 264 < y:
                    new_crop_pos = (x + crop_dim_x, y - crop_dim_y)
                else:
                    new_crop_pos = (x + crop_dim_x, y + crop_dim_y)

                crop2 = crop_one(img, crop_dim, new_crop_pos)

                yield crop, crop2, 1
            except ValueError:
                pass


def create_training_set(data_path, rel_pos):
    """
    Creates 3 arrays containing image crops and whether they are adjacent or not
    :param data_path: path relative to the directory of this script
    :return: the 3 arrays
    """

    curr_dir = os.path.dirname(__file__) #slightly strange
    img_dir = os.path.realpath(os.path.join(curr_dir, data_path))

    X1s = []
    X2s = []
    Ys = []

    select_correct_crops = {
        RelativePosition.RIGHT: select_correct_crops_right,
        RelativePosition.DOWN: select_correct_crops_down,
    }

    images = glob.glob(img_dir + '/**/*.jpg', recursive=True)
    for image_path in tqdm.tqdm(images, total=len(images)):

        img = Image.open(image_path)
        img = resizeimage.resize_cover(img, [528, 528])
        img = np.array(img)

        for X_1, X_2, Y in (select_correct_crops[rel_pos](img)):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)
        for X_1, X_2, Y in select_incorrect_crops(img):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)

    return X1s, X2s, Ys


# Layer names are : blockN_convM for N = [1,2,3,4,5] M=[1,2,3] except in block 1 and 2 where M=[1,2]

class FeatureExtraction:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)
        # layer_name="block2_pool"
        # self.model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer_name).output)

    def _feature_extraction(self, img):
        """
        Keras feature extraction model
        :param img:
        :return:
        """
        # img = image.load_img(img_path, target_size=(48, 48))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def _merge_features(self, features1, features2):
        """
        Merges the features of 2 image crops in one array
        :param features1:
        :param features2:
        :return:
        """
        features1 = np.reshape(features1, [-1])
        features2 = np.reshape(features2, [-1])
        res = np.concatenate((features1, features2))
        return res

    def extract_feature_pair(self, img1, img2):
        """
        What does this do?
        :param img1:
        :param img2:
        :return:
        """
        features_im1 = self._feature_extraction(img1)
        features_im2 = self._feature_extraction(img2)
        r = self._merge_features(features_im1, features_im2)
        return r

    def extract_concat_feature(self, img1, img2):
        img = np.concatenate((img1, img2), axis=0)
        feat = self._feature_extraction(img)
        return np.reshape(feat, [-1])

    def extract_feats_from_list(self, list):
        return [np.reshape(self._feature_extraction(file), [1, -1]) for file in list]

    def __del__(self):
        del self.model


def generate_dataset(image_path, save_path, rel_pos):
    """
    Creates a pickle file of images in a specified path
    :param image_path:
    :param save_path:
    :return:
    """
    feats = FeatureExtraction()
    training_set = create_training_set(image_path, rel_pos)
    features = []
    Ys = []

    for im1, im2, Y in tqdm.tqdm(zip(*training_set), total=len(training_set[-1])):
        try:
            # features.append(feats.extract_feature_pair(im1, im2))  # why are we not using merge features?
            features.append(feats.extract_concat_feature(im1, im2))
            Ys.append(Y)
        except(OSError):
            print("Corrupted file.... :(")

    features = np.array(features)
    Ys = np.array(Ys)

    with open(save_path, mode="wb") as fp:
        pickle.dump((features, Ys), fp)


if __name__ == "__main__":
    args = [
        ('../../../training_images', 'dataset_right.pkl', RelativePosition.RIGHT),
        ('../../../test_set', 'datatest_right.pkl', RelativePosition.RIGHT),
        ('../../../training_images', 'dataset_down.pkl', RelativePosition.DOWN),
        ('../../../test_set', 'datatest_down.pkl', RelativePosition.DOWN)
    ]

    jobs = []
    for arg in args:
        p = Process(target=generate_dataset, args=arg)
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()
