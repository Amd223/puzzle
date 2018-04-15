import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from puzzle.training_classifiers.extractors.feature_extractor import FeatureExtractor


# Layer names are : blockN_convM for N = [1,2,3,4,5] M=[1,2,3] except in block 1 and 2 where M=[1,2]


class VGG16FeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)
        # layer_name="block2_pool"
        # self.model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer(layer_name).output)

    @staticmethod
    def name():
        return 'vgg16'

    def _feature_extraction(self, img):
        """
        Keras feature extraction model
        :param img:
        :return:
        """
        # img = image.load_img(img_path, target_size=(48, 48))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def _merge_features(self, features1, features2):
        """
        Merges the features of 2 image crops in one array
        :param features1:
        :param features2:
        :return:
        """
        features1 = np.reshape(features1, [-1])
        features2 = np.reshape(features2, [-1])
        res = np.concatenate((features1, features2))
        return res

    def extract(self, img1, img2, is_down=True, **kwargs):
        if is_down:
            img = np.concatenate((img1, img2), axis=0)
        else:
            img = np.concatenate((img1, img2), axis=1)
        feat = self._feature_extraction(img)
        r = np.reshape(feat, [-1])
        print(img1.shape, img2.shape, r.shape)
        return r

    def extract_feats_from_list(self, list):
        return [np.reshape(self._feature_extraction(file), [1, -1]) for file in list]

    def __del__(self):
        del self.model