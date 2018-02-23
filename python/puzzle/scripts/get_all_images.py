import os
import glob
import tempfile
from random import randint

from puzzle.tools.crop import crop_loulou
from puzzle.tools.utils import input_directory, img_read
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tqdm
import pickle


def crop_one(img_path, crop_dim, crop_pos=(0, 0), save=True):
    """
    Extracts a crop from a given image
    :param img_path: str
        Path to the image to crop from
    :param crop_dim: (int, int)
        Dimensions (width, height) of the crop
    :param crop_pos: (int, int)
        Position of the top-left corner (x, y) of the crop
    :return: str, path of the directory containing the cropped images.
    """
    crop_width, crop_height = crop_dim
    crop_x, crop_y = crop_pos

    # Load image
    img = img_read(img_path)

    crop_img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    if not save:
        return crop_img

    # Create temp dir of outputs
    tmp_dir = tempfile.mkdtemp()
    _, img_extension = os.path.splitext(img_path)
    crop_name = os.path.join(tmp_dir, 'crop{}'.format(img_extension))
    cv2.imwrite(crop_name, crop_img)

    return crop_name


X_1 = []
X_2 = []
Y = []


def select_correct_crops(img_path, crop_dim=(16, 16), list1=X_1, list2=X_2, class_list=Y):
    x = randint(1, 15)
    y = randint(1, 16)

    crop_pos = (x, y)
    crop = crop_one(img_path, crop_dim, crop_pos, save=True)

    new_crop_pos = (x + 1, y)
    crop2 = crop_one(img_path, crop_dim, new_crop_pos, save=True)

    return crop, crop2, 0


def select_incorrect_crops(img_path, crop_dim=(16, 16), list1=X_1, list2=X_2, class_list=Y):
    x = randint(1, 13)
    y = randint(1, 13)

    crop_pos = (x, y)
    crop = crop_one(img_path, crop_dim, crop_pos, save=True)

    new_crop_pos = (x + 3, y + 3)
    crop2 = crop_one(img_path, crop_dim, new_crop_pos, save=True)
    return crop, crop2, 1


def create_training_set():
    curr_dir = os.path.dirname(__file__)
    img_dir = os.path.realpath(os.path.join(curr_dir, '../../../images'))

    X1s = []
    X2s = []
    Ys = []

    for image in glob.iglob(img_dir + '/**/*.jpg', recursive=True):
        X_1, X_2, Y = select_correct_crops(image)
        X1s.append(X_1)
        X2s.append(X_2)
        Ys.append(Y)
        X_1, X_2, Y = select_incorrect_crops(image)
        X1s.append(X_1)
        X2s.append(X_2)
        Ys.append(Y)

    return X1s, X2s, Ys


class FeatureExtraction:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    def _feature_extraction(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def _merge_features(self, features1, features2):
        features1 = np.reshape(features1, [-1])
        features2 = np.reshape(features2, [-1])
        return np.concatenate((features1, features2))

    def __call__(self, img_path1, img_path2):
        features_im1 = self._feature_extraction(img_path1)
        features_im2 = self._feature_extraction(img_path2)
        return self._merge_features(features_im1, features_im2)


if __name__ == "__main__":

    dataset_file = "dataset.pkl"
    feats = FeatureExtraction()
    training_set = create_training_set()
    features = []
    Ys = []
    for im1, im2, Y in tqdm.tqdm(zip(*training_set), total=len(training_set[-1])):
        try:
            features.append(feats(im1, im2))
            Ys.append(Y)
        except(OSError):
            print("Corrupted file.... :(")

    features = np.array(features)
    Ys = np.array(Ys)


    with open(dataset_file, mode="wb") as fp:
        pickle.dump((features, Ys), fp)

# if __name__ == "__main__":
#     curr_dir = os.path.dirname(__file__)
#     img_dir = os.path.realpath(os.path.join(curr_dir, '../../../images'))
#
#     for image in glob.iglob(img_dir + '/**/*.jpg', recursive=True):
#         select_correct_crops(image)
#
#     #cv2.imshow("img", select_crops(dir_in)[0])
#     #cv2.waitKey(0)
#     #cv2.destroyAllWindows()
