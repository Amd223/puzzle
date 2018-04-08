import numpy as np

from puzzle.scripts.reconstruction import crop_puzzle, evaluate_pieces, load_rfc
from puzzle.scripts.get_all_images import FeatureExtraction
from puzzle.tools.utils import input_directory
import matplotlib.pyplot as plt


def score_function_cnn(img1, img2):

    feat_extractor = FeatureExtraction()

    rfc = load_rfc("rfc.pkl")


#test_set/cities/city1.jpg

def reconstruct(img_path):
    """
    Take an image path, crop puzzle, reconstruct
    :param img_path:
    :return: reconstructed image
    """

    pieces = crop_puzzle(img_path)

    feat_extractor = FeatureExtraction()
    features = feat_extractor.extract_feats_from_list(pieces)

    rfc = load_rfc("rfc.pkl")

    probabilities = np.zeros((len(pieces), len(pieces)))
    probabilities2 = np.zeros((len(pieces), len(pieces)))

    for i, feats_left in enumerate(features):  # i is index of feats left in features. eg. features[i] == feats_left
        for j, feats_right in enumerate(features):
            print(feats_left.shape, feats_right.shape)
            if i != j:
                # probabilities[i, j] = 1/np.sqrt(np.sum(np.square(pieces[i][:,-1,:] - pieces[j][:,0,:])))
                probabilities[i,j] = rfc.predict_proba(np.concatenate((feats_left, feats_right), axis=-1))[0][0]

    left, right = np.unravel_index(np.argmax(probabilities, axis=None), probabilities.shape)
    left = 0
    combined_image = pieces[left]
    probabilities[:, left] = 0
    for _ in range(10):
        right = np.argmax(probabilities[left, :], axis=None)
        combined_image = np.concatenate((combined_image, pieces[right]), axis=1)
        left = right
        probabilities[:,right] = 0

    import pickle

    with open("for_ben.pkl", "wb") as fp:
        pickle.dump((probabilities, probabilities2, pieces), fp)







    plt.imshow(combined_image)
    plt.show()


if __name__ == "__main__":
    img_path = input_directory()
    reconstruct(img_path)
