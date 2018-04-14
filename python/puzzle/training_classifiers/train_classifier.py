import os
import pickle

import multiprocessing
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC

from puzzle.data_collection.create_sets import RelativePosition
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


def extract_buggy_pickle(x, y):
    # If pickle is fine
    if len(y) > 0:
        return x, y

    xx = []
    yy = []
    it = (e for e in x)
    try:
        while True:
            xx.append(next(it))
            yy.append(next(it))
    except StopIteration:
        pass

    return np.array(xx), np.array(yy)


def train_classifiers(feature, image_class=None, do_plot=True, display=True):
    class_name = image_class if image_class is not None else 'all'

    project_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))

    rel_positions = [p.value for p in RelativePosition]
    for rel_pos in rel_positions:

        def mkd(pattern):
            return os.path.join(project_base, pattern.format(class_name, feature, rel_pos))

        train_set_path = mkd('extracted_features/{}-{}-train-{}.pkl')
        test_set_path  = mkd('extracted_features/{}-{}-test-{}.pkl')

        print('Using training set ' + train_set_path)
        with open(train_set_path, mode="rb") as fp:
            # TODO: remove buggy pickle patch when pickle will be reconstructed
            # x_train, y_train = pickle.load(fp)
            x_train, y_train = extract_buggy_pickle(*pickle.load(fp))

        print('Using testing set ' + test_set_path)
        with open(test_set_path, mode="rb") as fp:
            # TODO: remove buggy pickle patch when pickle will be reconstructed
            # x_test, y_test = pickle.load(fp)
             x_test, y_test = extract_buggy_pickle(*pickle.load(fp))

        # Training classifiers
        lregression = LogisticRegression()
        svm = SVC()
        linsvm = LinearSVC()
        knn = KNeighborsClassifier()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        rfc = RandomForestClassifier()
        classifiers = [lregression, svm, linsvm, knn, clf, rfc]

        def worker(classifier, return_dict):
            '''worker function'''
            classifier.fit(x_train, y_train)
            return_dict[classifier] = classifier.score(x_test, y_test)

        manager = multiprocessing.Manager()
        scores = manager.dict()
        jobs = []

        for c in classifiers:
            p = multiprocessing.Process(target=worker, args=(c, scores))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        print('Classifier results {}-{}-{}'.format(class_name, feature, rel_pos))
        for c, score in scores:
            print('   >>> {} : {}'.format(c, score))

        # Saving to pickle file...
        save_path = mkd('trained_classifiers/rfc-{}-{}-{}.pkl')

        # Creates the necessary directories
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print('Saving RFC at ' + save_path)
        with open(save_path, mode="wb") as fp:
            pickle.dump(rfc, fp)

        if not do_plot:
            return

        # Plotting...
        y = label_binarize(y_test, classes=[0, 1])

        pred_lregression = lregression.decision_function(x_test)
        fpr_lregression, tpr_lregression, thresholds = roc_curve(y, pred_lregression)
        roc_auc_lregression = auc(fpr_lregression, tpr_lregression)

        pred_svm = svm.decision_function(x_test)
        fpr_svm, tpr_svm, thresholds = roc_curve(y, pred_svm)
        roc_auc_svm = auc(fpr_svm, tpr_svm)

        pred_linsvm = linsvm.decision_function(x_test)
        fpr_linsvm, tpr_linsvm, thresholds = roc_curve(y, pred_linsvm)
        roc_auc_linsvm = auc(fpr_linsvm, tpr_linsvm)

        # pred_knn = knn.predict(x_test)
        # fpr_knn, tpr_knn, thresholds = roc_curve(y, pred_knn)
        # roc_auc_knn = auc(fpr_knn, tpr_knn)

        # pred_clf = clf.predict(x_test)
        # fpr_clf, tpr_clf, thresholds = roc_curve(y, pred_clf)
        # roc_auc_clf = auc(fpr_clf, tpr_clf)
        #
        # pred_rfc = rfc.predict(x_test)
        # fpr_rfc, tpr_rfc, thresholds = roc_curve(y, pred_rfc)
        # roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
        # obtenir les scores
        # 2 variables en output?

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr_lregression, tpr_lregression, color='cornflowerblue', label='ROC curve lregression (area = %0.2f)' % roc_auc_lregression)
        plt.plot(fpr_svm, tpr_svm, color='darkorange', label='ROC curve SVM (area = %0.2f)' % roc_auc_svm)
        plt.plot(fpr_linsvm, tpr_linsvm, color='green', label='ROC curve LINSVM (area = %0.2f)' % roc_auc_linsvm)
        # plt.plot(fpr_knn, tpr_knn, color='aqua', label='ROC curve KNN (area = %0.2f)' % roc_auc_knn)
        # plt.plot(fpr_clf, tpr_clf, color='black', label='ROC curve clf (area = %0.2f)' % roc_auc_clf)
        # plt.plot(fpr_rfc, tpr_rfc, color='darkgreen', label='ROC curve RFC (area = %0.2f)' % roc_auc_rfc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        if display:
            plt.show()
        else:
            plt.savefig(mkd('trained_classifiers/rfc-{}-{}-{}.png'))


if __name__ == "__main__":

    jobs = []
    for f in [VGG16FeatureExtractor.name()]:
        for c in ['animals', 'art', 'cities', 'landscapes', 'portraits', 'space', None]:
            # train_classifiers(f, c, do_plot=True, display=True)

            print('\nTraining classifier for {}...'.format(c))
            p = multiprocessing.Process(target=train_classifiers, args=(f, c), kwargs=dict(do_plot=False, display=True))
            jobs.append(p)
            p.start()

    for p in jobs:
        p.join()
