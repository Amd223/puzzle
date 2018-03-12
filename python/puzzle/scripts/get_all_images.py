import os
import glob
import tempfile
from random import randint
from PIL import Image
from resizeimage import resizeimage
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tqdm
import pickle


def crop_one(img_path, crop_dim, crop_pos=(0, 0), save=False):
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

    img = Image.open(img_path)
    img = resizeimage.resize_cover(img, [528, 528])
    img = np.array(img)

    crop_img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    if crop_img.shape[:2] != crop_dim:

        raise ValueError("Image crop not square, has dims:{}, at location:{}".format(crop_img.shape[:2], crop_pos))

    if not save:
        return crop_img

    # # Create temp dir of outputs
    # tmp_dir = tempfile.mkdtemp()
    # _, img_extension = os.path.splitext(img_path)
    # crop_name = os.path.join(tmp_dir, 'crop{}'.format(img_extension))
    # cv2.imwrite(crop_name, crop_img)
    #
    # return crop_name # how to find this tmp dir?



def select_correct_crops(img_path, crop_dim=(48, 48)):
    """
      Extracts x amount of correct crop pairs from a given image
      :param img_path: str
          Path to the image to crop from
      :param crop_dim: (int, int)
          Dimensions (width, height) of the crop
    """

    for x in range(0, 528, 48*2):
        for y in range(0, 528, 48*2):
            try:
                crop_pos = (x, y)
                crop = crop_one(img_path, crop_dim, crop_pos, save=False)

                new_crop_pos = (x + crop_dim[0], y)
                crop2 = crop_one(img_path, crop_dim, new_crop_pos, save=False)

                yield crop, crop2, 0 #what does yield do?

            except ValueError:
                pass


def select_incorrect_crops(img_path, crop_dim=(48, 48)):
    """
          Extracts x amount of incorrect crop pairs from a given image
          :param img_path: str
              Path to the image to crop from
          :param crop_dim: (int, int)
              Dimensions (width, height) of the crop
          """

    for x in range(0, 528, 48*2):  # why only 0-15?
        for y in range(0, 528, 48*2):
            try:
                # if the crop is not a square, pass and don't put in dataset
                crop_pos = (x, y)
                crop = crop_one(img_path, crop_dim, crop_pos, save=False)

                if (x,y) > (264,264):
                    new_crop_pos = (x - crop_dim[0]*5, y - crop_dim[1]*5)
                elif x > 264 & y < 264:
                    new_crop_pos = (x - crop_dim[0]*5, y + crop_dim[1]*5)
                elif x < 264 & y > 264:
                    new_crop_pos = (x + crop_dim[0]*5, y - crop_dim[1]*5)
                if (x,y) < (264,264):
                    new_crop_pos = (x - crop_dim[0]*5, y - crop_dim[1]*5)


                crop2 = crop_one(img_path, crop_dim, new_crop_pos, save=False)

                yield crop, crop2, 1
            except ValueError:
                pass

def create_training_set(data_path):
    """
    Creates 3 arrays containing image crops and wether they are adjacent or not
    :param data_path:
    :return:
    """

    curr_dir = os.path.dirname(__file__)
    img_dir = os.path.realpath(os.path.join(curr_dir,data_path))

    X1s = []
    X2s = []
    Ys = []

    for image in tqdm.tqdm(glob.iglob(img_dir + '/**/*.jpg', recursive=True)):
        for X_1, X_2, Y in select_correct_crops(image):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)
        for X_1, X_2, Y in select_incorrect_crops(image):
            X1s.append(X_1)
            X2s.append(X_2)
            Ys.append(Y)

    return X1s, X2s, Ys


class FeatureExtraction:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    def _feature_extraction(self, img_path):
        """
        Keras feature extraction model
        :param img_path:
        :return:
        """
        #img = image.load_img(img_path, target_size=(48, 48))
        x = image.img_to_array(img_path)
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
        return np.concatenate((features1, features2))

    def extract_feature_pair(self, img_path1, img_path2):
        """
        What does this do?
        :param img_path1:
        :param img_path2:
        :return:
        """
        features_im1 = self._feature_extraction(img_path1)
        features_im2 = self._feature_extraction(img_path2)
        return self._merge_features(features_im1, features_im2)

    def extract_feats_from_list(self, list):
        feats = []
        for file in list:
            feats.append(np.reshape(self._feature_extraction(file), [1, -1]))
        return feats

    def __del__(self):
        del self.model

def generate_dataset(image_path, save_path):
    """
    Creates a pickle file of images in a specified path
    :param image_path:
    :param save_path:
    :return:
    """
    feats = FeatureExtraction()
    training_set = create_training_set(image_path)
    features = []
    Ys = []
    for im1, im2, Y in tqdm.tqdm(zip(*training_set), total=len(training_set[-1])):
        try:
            features.append(feats.extract_feature_pair(im1, im2)) # why are we not using merge features?
            Ys.append(Y)
        except(OSError):
            print("Corrupted file.... :(")

    features = np.array(features)
    Ys = np.array(Ys)

    with open(save_path, mode="wb") as fp:
        pickle.dump((features, Ys), fp)


if __name__ == "__main__":
    dataset_file = "dataset.pkl"
    generate_dataset('../../../training_images', dataset_file)
    datatest_file = "datatest.pkl"
    generate_dataset('../../../test_set', datatest_file)

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
