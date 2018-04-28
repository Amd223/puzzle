import argparse
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


def load_test_image(image_class, show=False, img_nb=0):
    _, test_set = get_image_sets(image_class=image_class)
    file_path = test_set[img_nb]
    print('Loading puzzle image ' + file_path)
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
        raise RuntimeError('Expected 1, but found #{} classifiers for pattern: {}'.format(len(file_names), file_pattern))
    filename = file_names[0]

    print('Using classifier {}'.format(os.path.basename(filename)))
    with open(filename, mode="rb") as fp:
        return ClassifierWrapper(pickle.load(fp))


def reconstruct_puzzle(image, classifier_down, classifier_right, feature_extractor, save_name, piece_size=(48, 48), display=True):
    piece_height, piece_width = piece_size
    height, width = image.shape[:2]

    # pieces of format { <location>: <image> }
    pieces = break_into_pieces(image, piece_size)

    # Perform reconstruction
    errors = 0
    rel_errors = 0
    reconstructed_puzzle = np.zeros(image.shape, dtype=np.uint8)

    remaining_pieces = list(pieces.values())

    top_left = pieces[(0, 0)]
    remaining_pieces = [p for p in remaining_pieces if not np.array_equal(p, top_left)]

    # Assume we now how to find the top-left corner
    random.shuffle(remaining_pieces)
    reconstructed_puzzle[0:piece_height, 0:piece_width, :] = top_left

    try:
        for y in range(0, height, piece_height):
            for x in range(0, width, piece_width):
                # Skip top_left
                if x == 0 and y == 0:
                    continue

                img_up =   reconstructed_puzzle[y - piece_height:y, x:x + piece_width]
                img_left = reconstructed_puzzle[y:y + piece_height, x - piece_width:x]

                classifying_triple = []

                # If there's a valid image up, use info from classifier_down
                if img_up.shape[:2] == piece_size:
                    classifying_triple.append((classifier_down, img_up, True))

                # If there's a valid image left, use info from classifier_right
                if img_left.shape[:2] == piece_size:
                    classifying_triple.append((classifier_right, img_left, False))

                piece, prob = predict_piece(classifying_triple, feature_extractor, remaining_pieces)

                remaining_pieces = [p for p in remaining_pieces if not np.array_equal(p, piece)]
                reconstructed_puzzle[y:y+piece_height, x:x+piece_width, :] = piece

                # Record errors
                selected_y, selected_x = get_coord_for_piece(pieces, piece)
                if (selected_y, selected_x) != (y, x):
                    errors += 1

                # Relative accuracy
                if x == 0:
                    # Left-most column - compare with piece above
                    if get_coord_for_piece(pieces, img_up) != (selected_y - piece_height, selected_x):
                        rel_errors += 1
                else:
                    # In-row - compare with piece to the left
                    if get_coord_for_piece(pieces, img_left) != (selected_y, selected_x - piece_width):
                        rel_errors += 1

                print('[index=({},{})] selected piece at index ({},{}) with proba {}'.format(y, x, selected_y, selected_x, prob))

    except Exception as e:
        print('Interrupted with error: {}'.format(e))

    correct_pieces = len(pieces) - errors
    accuracy = int(correct_pieces / len(pieces) * 100)
    accuracy = '{}% ({}/{})'.format(accuracy, correct_pieces, len(pieces))

    correct_pieces = len(pieces) - rel_errors
    rel_accuracy = int(correct_pieces / len(pieces) * 100)
    rel_accuracy = '{}% ({}/{})'.format(rel_accuracy, correct_pieces, len(pieces))

    print('Finished with accuracy={}, rel_accuracy={}'.format(accuracy, rel_accuracy))

    # Plotting
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_puzzle)
    plt.title(r"""Reconstructed 
    accuracy={}
    rel_accuracy={}""".format(accuracy, rel_accuracy))

    # Save image
    save_path = mk_path(os.path.join('reconstructed', save_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    if display:
        plt.show()


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


def predict_piece(classifying_triple, feature_extractor, remaining_pieces):
    if len(classifying_triple) == 0:
        raise RuntimeError('Need at least one classifying_triple')

    probabilities = np.ones(len(remaining_pieces))

    for clf, img_comp, is_down in classifying_triple:

        clf_probs = np.zeros(len(remaining_pieces))
        for i, piece in enumerate(remaining_pieces):
            features = feature_extractor.extract(img_comp, piece, is_down=is_down)
            clf_probs[i] = clf.get_proba_is_adjacent(features)

        probabilities = np.multiply(probabilities, clf_probs)

    # print('Probabilities: ', probabilities)
    index_best = np.argmax(probabilities)
    return remaining_pieces[index_best], probabilities[index_best]


def get_coord_for_piece(pieces, piece):
    for coordinate, p in pieces.items():
        if np.array_equal(p, piece):
            return coordinate


def do_reconstruction(image_class_name=None, feature='vgg16', img_idx=0, img_size=11, show_in=False, show_out=True):
    image_class = None if image_class_name == 'all' else image_class_name
    assert 1 <= img_size <= 11, 'Image side size must be in the range 1-11'

    # Init image for puzzle
    image = load_test_image(image_class, show=show_in, img_nb=img_idx)

    # Init feature extractor
    feature_extractors = {
        VGG16FeatureExtractor.name(): VGG16FeatureExtractor,
        L2FeatureExtractor.name(): L2FeatureExtractor,
        ColourFeatureExtractor.name(): ColourFeatureExtractor,
        GradientFeatureExtractor.name(): GradientFeatureExtractor,
    }
    feature_extractor = feature_extractors[feature]()

    # Init classifiers
    classifier_down, classifier_right = load_classifier_pair(image_class_name, feature)

    # Reconstruct
    image = image[:48*img_size, :48*img_size, :]
    save_name = 'rec-{}-n{}-{}pcs-{}'.format(image_class_name, img_idx, img_size**2, feature)
    reconstruct_puzzle(image, classifier_down, classifier_right, feature_extractor, save_name, display=show_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='image_class_name', help='name of image class to use. Default is all.',
                        default='all', type=str)
    parser.add_argument('-f', dest='feature', help='feature type to use. Default is vgg16.',
                        default='vgg16', type=str)
    parser.add_argument('-n', dest='img_idx', help='index of image in test set. Default is 0.', default=0, type=int)
    parser.add_argument('-s', dest='img_size', help='number of pieces per side - range 1-11. Default is 11.',
                        default=11, type=int)
    parser.add_argument('--show-input', dest='show_in', action='store_true', default=False,
                        help='show input image. Default is False.')
    parser.add_argument('--show-output', dest='show_out', action='store_true', default=False,
                        help='show reconstructed image. Default is False.')
    args = parser.parse_args()

    do_reconstruction(**vars(args))
