import os
import glob
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from puzzle.data_collection.create_sets import get_image_sets
from puzzle.training_classifiers.classifier_wrapper import ClassifierWrapper
from puzzle.training_classifiers.extractors.l2_features import L2FeatureExtractor


def reconstruct_puzzle(image, classifier_down, classifier_right, feature_extractor, piece_size=(48, 48), display=True):
    piece_height, piece_width = piece_size
    height, width = image.shape[:2]

    # pieces of format { <location>: <image> }
    pieces = break_into_pieces(image, piece_size)

    # Perform reconstruction
    errors = 0
    reconstructed_puzzle = np.zeros((height, width))

    remaining_pieces = list(pieces.values())
    random.shuffle(remaining_pieces)
    top_left = remaining_pieces.pop(0)

    # Assume we now how to find the top-left corner
    reconstructed_puzzle[0:piece_height, 0:piece_width] = top_left

    for y in range(0, height, piece_height):
        for x in range(0, width, piece_width):
            # Skip top_left
            if x == 0 and y == 0:
                continue

            if x == 0:
                # Use down_classifier to place first of row
                imp_up = reconstructed_puzzle[piece_height-y:y, x:x+piece_width]
                piece, prob = predict_piece(classifier_down, feature_extractor, imp_up, remaining_pieces, is_down=True)
            else:
                # Use right_classifier to place the rest of row
                img_left = reconstructed_puzzle[y:y+piece_height, x-piece_width:x]
                piece, prob = predict_piece(classifier_right, feature_extractor, img_left, remaining_pieces, is_down=False)

            reconstructed_puzzle[y:y+piece_height,x:x+piece_width] = piece

            # Record errors
            selected_coord = get_coord_for_piece(pieces, piece)
            if selected_coord != (y, x):
                errors += 1

            print('[index=({},{})] selected piece at index ({},{}) with proba {}'.format(y, x, *selected_coord, prob))

    plt.imshow(reconstructed_puzzle)
    if display:
        plt.show()
    else:
        plt.savefig(mk_path('reconstructed.png'))

    print('Finished with errors: %d%% (%d/%d)' % (errors / len(pieces) * 100, errors, len(pieces)))


def load_classifier_pair(*args):
    return load_classifier(*args, 'down'), load_classifier(*args, 'right')


def mk_path(path):
    project_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))
    return os.path.join(project_base, path)


def load_classifier(class_name, feature, rel_pos):
    file_pattern = mk_path('trained_classifiers/{}-{}-{}-*.pkl'.format(class_name, feature, rel_pos))
    filename = glob.glob(file_pattern)[0]

    with open(filename, mode="rb") as fp:
        return ClassifierWrapper(pickle.load(fp))


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
        features = feature_extractor.extract(img_comp, piece, is_down=is_down).reshape([1, -1])
        probabilities[i] = classifier.predict(features)[0][0]

    proba_best = np.argmax(probabilities, axis=None)
    index_best = np.unravel_index(proba_best, probabilities.shape)
    return remaining_pieces[index_best], proba_best


def get_coord_for_piece(pieces, piece):
    for coordinate, p in pieces.items():
        if p == piece:
            return coordinate


def do_reconstruction():
    _, test_set = get_image_sets(image_class='animals')
    image = np.array(Image.open(test_set[0]))

    classifier_down, classifier_right = load_classifier_pair('animals', 'L2')
    l2 = L2FeatureExtractor()
    reconstruct_puzzle(image, classifier_down, classifier_right, l2, display=False)


if __name__ == "__main__":
    do_reconstruction()