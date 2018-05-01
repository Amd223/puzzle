import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


class VGG16FeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    @staticmethod
    def name():
        return 'vgg16'

    def _feature_extraction(self, img):
        """
        Keras feature extraction model
        :param img:
        :return:
        """
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def extract(self, img1, img2, is_down=True, **kwargs):
        if is_down:
            img = np.concatenate((img1, img2), axis=0)
        else:
            img = np.concatenate((img1, img2), axis=1)
        feat = self._feature_extraction(img)
        return np.reshape(feat, [-1])

    def __del__(self):
        del self.model