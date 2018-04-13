class FeatureExtractor:
    def name(self):
        raise NotImplementedError('Need to implement this')

    def extract(self, img1, img2):
        raise NotImplementedError('Need to implement this')

    def extract_feats_from_list(self, list):
        raise NotImplementedError('Need to implement this')