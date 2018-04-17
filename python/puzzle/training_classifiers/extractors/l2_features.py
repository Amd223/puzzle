import numpy as np

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class L2FeatureExtractor(FeatureExtractor):
    @staticmethod
    def name():
        return 'L2'

    def extract(self, img1, img2, **kwargs):
        # d = np.sqrt(np.sum(np.square(img1 - img2), axis=1))[:, None, None]
        assert img1.shape == img2.shape, "Expect crop shapes to be equal"

        # Normalise image shapes. Shapes encountered:
        #   - grey scale: (48, 48)
        #   - RGB:        (48, 48, 3)
        #   - RGBA:       (48, 48, 4)
        if len(img1.shape) >= 3:
            # Sum squared difference of three values
            x = np.sum(np.square(img1[:, :, :3] - img2[:, :, :3]), axis=2)
        else:
            # Squared difference, times 3 to have the same magnitude as above
            x = np.square(img1 - img2) * 3

        # Reshape in blocks of 2 rows (24, 96)
        side = img1.shape[0]
        x = np.reshape(x, (-1, 2 * side))

        # Group each line by group of 4 -> (24, 24)
        f = lambda x: x\
            .reshape((2, -1))\
            .sum(axis=0)\
            .reshape((-1, 2))\
            .sum(axis=1)
        x = np.apply_along_axis(f, 1, x)

        # Flatten features
        return np.reshape(x, -1)

    # def extract_feats_from_list(self, list):
    #     pass
