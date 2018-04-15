from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class GradientFeatureExtractor(FeatureExtractor):
    @staticmethod
    def name():
        return 'hist-gradient'

    def extract(self, img1, img2, **kwargs):
        pass

    def extract_feats_from_list(self, list):
        pass
