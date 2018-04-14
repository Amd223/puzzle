import os
import tqdm
import pickle
import numpy as np
from multiprocessing import Process

from puzzle.data_collection.create_sets import get_sets
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


def generate_extracted_features(image_set, featureExtractor, pickle_name):
    """
    Creates a pickle file of images in a specified path
    """
    # Construct path to save pickle file to
    curr_dir    = os.path.dirname(__file__)
    pickle_file = '../../../extracted_features/{}.pkl'.format(pickle_name)
    save_path   = os.path.realpath(os.path.join(curr_dir, pickle_file))

    # Creates the necessary directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Perform feature extraction
    features = []
    ys = []

    for im1, im2, y in tqdm.tqdm(zip(*image_set), total=len(image_set[0])):
        try:
            features.append(featureExtractor.extract(im1, im2))
            ys.append(y)
        except(Exception) as e:
            print('Error on sample: {}'.format(e))

    # Save results to pickle file
    features = np.array(features)
    ys       = np.array(ys)
    with open(save_path, mode="wb") as fp:
        pickle.dump((features, ys), fp)


def do_feature_extraction(feature_extractors, image_class=None):
    class_name = image_class if image_class is not None else 'all'

    sets_name = ['train-down', 'train-right', 'test-down', 'test-right']
    print('Get training / test sets for down / right positions')
    sets = get_sets(image_class, seed=42)

    for feature_extractor in feature_extractors:
        for set_name, image_set in zip(sets_name, sets):
            pickle_name = '{}-{}-{}'.format(class_name, feature_extractor.name(), set_name)

            print('Start feature extraction for ' + pickle_name)
            generate_extracted_features(image_set, feature_extractor, pickle_name)


if __name__ == "__main__":

    for c in ['animals', 'art', 'cities', 'landscapes', 'portraits', 'space', None]:
        print('\nExtracting {}...'.format(c))
        do_feature_extraction([
            VGG16FeatureExtractor()
        ], image_class=c)
