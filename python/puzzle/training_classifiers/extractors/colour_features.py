import cv2
import numpy as np

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class ColourFeatureExtractor(FeatureExtractor):
    @staticmethod
    def name():
        return 'hist-colour'

    def extract(self, img1, img2, **kwargs):
        no_features = 256

        diff = []
        for i, col in enumerate(['b', 'g', 'r']):
            hist1 = cv2.calcHist([img1], [i], None, [no_features], [0, no_features])
            hist2 = cv2.calcHist([img2], [i], None, [no_features], [0, no_features])
            diff.append(np.square(hist1 - hist2))

        return np.sqrt(np.sum(diff, axis=0)).reshape(-1)