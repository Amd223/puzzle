import os
import pickle

import tqdm
import numpy as np

from puzzle.data_collection.create_sets import get_sets
from puzzle.training_classifiers.extractors.colour_features import ColourFeatureExtractor
from puzzle.training_classifiers.extractors.gradient_features import GradientFeatureExtractor
from puzzle.training_classifiers.extractors.l2_features import L2FeatureExtractor
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


def generate_extracted_features(image_set, featureExtractor, pickle_name):
    """
    Creates a pickle file of images in a specified path
    """
    is_down = pickle_name.endswith('down')

    # Construct path to save pickle file to
    curr_dir    = os.path.dirname(__file__)
    pickle_file = '../../../extracted_features/{}.pkl'.format(pickle_name)
    save_path   = os.path.realpath(os.path.join(curr_dir, pickle_file))

    # Creates the necessary directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Perform feature extraction
    features = []
    ys = []

    errors = 0
    total = len(image_set[0])
    last_error = None

    for im1, im2, y in tqdm.tqdm(zip(*image_set), total=total):
        try:
            features.append(featureExtractor.extract(im1, im2, is_down=is_down))
            ys.append(y)
        except(Exception) as e:
            msg = str(e)
            if msg != last_error:
                print('Error on sample: ' + msg)
                last_error = msg
            errors += 1

    print('Finished with errors: %d%% (%d/%d)' % (errors / total * 100, errors, total))

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

            print('\nStart feature extraction for ' + pickle_name)
            generate_extracted_features(image_set, feature_extractor, pickle_name)


if __name__ == "__main__":

    for c in ['animals', 'art', 'cities', 'landscapes', 'portraits', 'space', None]:
        print('\nExtracting {}...'.format(c))
        do_feature_extraction([
            # VGG16FeatureExtractor(),
            # L2FeatureExtractor(),
            # ColourFeatureExtractor(),
            GradientFeatureExtractor()
        ], image_class=c)
