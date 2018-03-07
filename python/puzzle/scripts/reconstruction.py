# Take image, slice to pieces then extract features.
import numpy as np

from puzzle.scripts.get_all_images import crop_one, FeatureExtraction
from PIL import Image
from puzzle.tools.utils import input_directory
from PIL import Image
import pickle


def crop_puzzle(img_path, crop_dimensions=(48,48)):
    """
    Crops an image into equally sized pieces
    :param img_path:
    :param crop_dimensions:
    :return:
    """
    X = []
    for i in range(0, 512, 48):
        for j in range(0, 512, 48):
            filename = crop_one(img_path, crop_dimensions, crop_pos=(i,j), save=True)
            X.append(filename)
    return X

def reconstruct(img_path):
    """
    Reconstructs an original image from a set of image pieces
    :param img_path:
    :return:
    """

    filenames = crop_puzzle(img_path, (48, 48))

    feat_extractor = FeatureExtraction()
    features = feat_extractor.extract_feats_from_list(filenames)

    for i in range(len(features)):
        print(evaluate_pieces(rfc, features[i], features[i + 1]))

def evaluate_pieces(rfc, feat1, feat2):
    """
    Asses whether 2 pieces should be adjacent or not
    :param crop1:
    :param crop2:
    :return:
    """
    score = rfc.predict(np.concatenate((feat1, feat2), axis=-1))

    if score[0] >= 0.6:
        return True
    else:
        return False

def load_rfc(filename):
    with open(filename, mode="rb") as fp:
        return pickle.load(fp)



if __name__ == "__main__":
    rfc = load_rfc("rfc.pkl")
    img_path = input_directory()
    filenames = crop_puzzle(img_path, (48,48))

    feat_extractor = FeatureExtraction()
    features = feat_extractor.extract_feats_from_list(filenames)

    for i in range(len(features)):
        print(evaluate_pieces(rfc, features[i], features[i+1]))







