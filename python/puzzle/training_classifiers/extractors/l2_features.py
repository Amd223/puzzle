import numpy as np

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class L2FeatureExtractor(FeatureExtractor):
    @staticmethod
    def name():
        return 'L2'

    def extract(self, img1, img2, **kwargs):
        return np.sqrt(np.sum(np.square(img1 - img2), axis=1))[:, None, None]

    def extract_feats_from_list(self, list):
        pass
