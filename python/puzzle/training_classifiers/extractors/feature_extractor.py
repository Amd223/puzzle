class FeatureExtractor:
    @staticmethod
    def name():
        raise NotImplementedError('Need to implement this')

    def extract(self, img1, img2, **kwargs):
        raise NotImplementedError('Need to implement this')