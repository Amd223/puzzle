import cv2
import numpy as np

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class GradientFeatureExtractor(FeatureExtractor):
    @staticmethod
    def name():
        return 'hist-gradient'

    def extract(self, img1, img2, **kwargs):
        """
        Source: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
        :return:
        """
        hist1 = self._compute_hist_of_gradients(img1)
        hist2 = self._compute_hist_of_gradients(img2)
        return np.square(hist1 - hist2)

    def _compute_hist_of_gradients(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16  # Number of bins
        bin = np.int32(bin_n * ang / (2 * np.pi))

        bin_cells = []
        mag_cells = []

        cellx = celly = 8

        dim_y, dim_x = img.shape[:2]
        for i in range(0, dim_y, celly):
            for j in range(0, dim_x, cellx):
                bin_cells.append(bin[i:i+celly, j:j+cellx])
                mag_cells.append(mag[i:i+celly, j:j+cellx])

        hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
                 for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= np.linalg.norm(hist) + eps

        return hist
