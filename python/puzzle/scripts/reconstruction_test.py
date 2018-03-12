from puzzle.scripts.reconstruction import crop_puzzle, evaluate_pieces, load_rfc
from puzzle.scripts.get_all_images import FeatureExtraction
from puzzle.tools.utils import input_directory


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






if __name__ == "__main__":
    img_path = input_directory()
    reconstruct(img_path)
