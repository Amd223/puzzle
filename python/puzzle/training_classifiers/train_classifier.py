import pickle

import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC

from puzzle.data_collection.create_sets import RelativePosition
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


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
            x_train, y_train = pickle.load(fp)

        print('Using testing set ' + test_set_path)
        with open(test_set_path, mode="rb") as fp:
            x_test, y_test = pickle.load(fp)

        # Training classifiers
        lregression = LogisticRegression()
        lregression.fit(x_train, y_train)
        print(lregression.score(x_test, y_test))

        svm = SVC()
        svm.fit(x_train, y_train)
        print(svm.score(x_test, y_test))

        linsvm = LinearSVC()
        linsvm.fit(x_train, y_train)
        print(linsvm.score(x_test, y_test))

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        print(knn.score(x_test, y_test))

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))

        rfc = RandomForestClassifier()
        rfc.fit(x_train, y_train)
        print(rfc.score(x_test, y_test))

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

    for f in [VGG16FeatureExtractor.name()]:
        for c in ['animals', 'art', 'cities', 'landscapes', 'portraits', 'space', None]:
            print('\nTraining classifier for {}...'.format(c))
            train_classifiers(f, c, do_plot=True, display=False)