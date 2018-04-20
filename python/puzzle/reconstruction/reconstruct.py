import os
import glob
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from puzzle.data_collection.create_sets import get_image_sets
from puzzle.training_classifiers.classifier_wrapper import ClassifierWrapper
from puzzle.training_classifiers.extractors.colour_features import ColourFeatureExtractor
from puzzle.training_classifiers.extractors.gradient_features import GradientFeatureExtractor
from puzzle.training_classifiers.extractors.l2_features import L2FeatureExtractor
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


def mk_path(path):
    project_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))
    return os.path.join(project_base, path)


def load_test_image(image_class, show=False):
    _, test_set = get_image_sets(image_class=image_class)
    file_path = test_set[2]
    image = np.array(Image.open(file_path))

    if show:
        plt.imshow(image)
        plt.show()

    return image


def load_classifier_pair(*args):
    return load_classifier(*args, 'down'), load_classifier(*args, 'right')


def load_classifier(image_class, feature, rel_pos):
    if image_class is None:
        image_class = 'all'

    file_pattern = mk_path('trained_classifiers/{}-{}-{}-*.pkl'.format(image_class, feature, rel_pos))
    file_names = glob.glob(file_pattern)
    if len(file_names) != 1:
        raise RuntimeError('Expected 1, but found #{} classifiers for pattern: {}'.format(len(filenames), file_pattern))
    filename = file_names[0]

    print('Using classifier {}'.format(os.path.basename(filename)))
    with open(filename, mode="rb") as fp:
        return ClassifierWrapper(pickle.load(fp))


def reconstruct_puzzle(image, classifier_down, classifier_right, feature_extractor, piece_size=(48, 48), display=True):
    piece_height, piece_width = piece_size
    height, width = image.shape[:2]

    # pieces of format { <location>: <image> }
    pieces = break_into_pieces(image, piece_size)

    # Perform reconstruction
    errors = 0
    reconstructed_puzzle = np.zeros(image.shape)

    remaining_pieces = list(pieces.values())
    top_left = remaining_pieces.pop(0)
    random.shuffle(remaining_pieces)

    # Assume we now how to find the top-left corner
    reconstructed_puzzle[0:piece_height, 0:piece_width, :] = top_left

    try:
        for y in range(0, height, piece_height):
            for x in range(0, width, piece_width):
                # Skip top_left
                if x == 0 and y == 0:
                    continue

                if x == 0:
                    # Use down_classifier to place first of row
                    imp_up = reconstructed_puzzle[y-piece_height:y, x:x+piece_width]
                    piece, prob = predict_piece(classifier_down, feature_extractor, imp_up, remaining_pieces, is_down=True)
                else:
                    # Use right_classifier to place the rest of row
                    img_left = reconstructed_puzzle[y:y+piece_height, x-piece_width:x]
                    piece, prob = predict_piece(classifier_right, feature_extractor, img_left, remaining_pieces, is_down=False)

                remaining_pieces = [p for p in remaining_pieces if not np.array_equal(p, piece)]
                reconstructed_puzzle[y:y+piece_height, x:x+piece_width, :] = piece

                # Record errors
                selected_coord = get_coord_for_piece(pieces, piece)
                if selected_coord != (y, x):
                    errors += 1

                print('[index=({},{})] selected piece at index ({},{}) with proba {}'.format(y, x, *selected_coord, prob))

    except Exception as e:
        print('Interrupted with error: {}'.format(e))

    plt.imshow(reconstructed_puzzle)
    if display:
        plt.show()
    else:
        plt.savefig(mk_path('reconstructed.png'))

    print('Finished with errors: %d%% (%d/%d)' % (errors / len(pieces) * 100, errors, len(pieces)))


def break_into_pieces(image, piece_size):
    piece_height, piece_width = piece_size
    height, width = image.shape[:2]

    pieces = {}
    for y in range(0, height, piece_height):
        for x in range(0, width, piece_width):
            coord = (y, x)
            piece = image[y:y + piece_height, x:x + piece_width]
            pieces[coord] = piece

    return pieces


def predict_piece(classifier, feature_extractor, img_comp, remaining_pieces, is_down=None):
    probabilities = np.zeros(len(remaining_pieces))

    for i, piece in enumerate(remaining_pieces):
        features = feature_extractor.extract(img_comp, piece, is_down=is_down).reshape(1, -1)
        probabilities[i] = classifier.predict(features)[0]

    print('Probabilities: ', probabilities)
    index_best = np.argmax(probabilities)
    return remaining_pieces[index_best], probabilities[index_best]


def get_coord_for_piece(pieces, piece):
    for coordinate, p in pieces.items():
        if np.array_equal(p, piece):
            return coordinate


def do_reconstruction(image_class, feature):
    # Init image for puzzle
    image = load_test_image(image_class, show=False)

    # Init feature extractor
    feature_extractors = {
        VGG16FeatureExtractor.name(): VGG16FeatureExtractor,
        L2FeatureExtractor.name(): L2FeatureExtractor,
        ColourFeatureExtractor.name(): ColourFeatureExtractor,
        GradientFeatureExtractor.name(): GradientFeatureExtractor,
    }
    feature_extractor = feature_extractors[feature]()

    # Init classifiers
    classifier_down, classifier_right = load_classifier_pair(image_class, feature)

    # Reconstruct
    reconstruct_puzzle(image, classifier_down, classifier_right, feature_extractor, display=True)


if __name__ == "__main__":
    do_reconstruction('portraits', 'L2')