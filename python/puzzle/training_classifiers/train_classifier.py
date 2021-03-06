import argparse
import multiprocessing
import operator
import os
import pickle

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from puzzle.data_collection.create_sets import RelativePosition
from puzzle.training_classifiers.classifier_wrapper import ClassifierWrapper
from puzzle.training_classifiers.extractors.colour_features import ColourFeatureExtractor
from puzzle.training_classifiers.extractors.gradient_features import GradientFeatureExtractor
from puzzle.training_classifiers.extractors.l2_features import L2FeatureExtractor
from puzzle.training_classifiers.extractors.vgg16_features import VGG16FeatureExtractor


def train_classifiers(rel_pos, feature, image_class=None, do_plot=True, save_plot_info=False, display=True):
    class_name = image_class if image_class is not None else 'all'

    project_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))

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
    classifiers = [
        # 'sag' solver for large data sets -
        # http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        LogisticRegression(solver='sag', n_jobs=-1),
        # LinearSVM - see ClassifierWrapper constructor for further details
        SVC(kernel='linear', probability=True),
        KNeighborsClassifier(n_jobs=-1),
        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
        RandomForestClassifier(n_jobs=-1)
    ]
    classifiers = [ClassifierWrapper(clf) for clf in classifiers]

    def worker(id, classifier, return_dict):
        """worker function"""
        start = time.time()
        cls_name = classifier.get_name()
        print('Starting {0: <22} on #features={1: <4}...'.format(cls_name, len(x_train[0])))
        classifier.fit(x_train, y_train)
        return_dict[id] = (classifier, classifier.score(x_test, y_test))
        print('Finished {0: <22} in {1:.2f}sec'.format(cls_name, time.time() - start))

    manager = multiprocessing.Manager()
    scores = manager.dict()
    jobs = []

    for i, c in enumerate(classifiers):
        p = multiprocessing.Process(target=worker, args=(i, c, scores))
        jobs.append(p)
        p.start()

    for i, p in enumerate(jobs):
        p.join()
        classifiers[i] = scores[i][0]

    # Saving to pickle file...
    best_classifier, best_score = max(scores.values(), key=operator.itemgetter(1))
    best_classifier_name = best_classifier.get_name()

    save_path = mkd('trained_classifiers/{}-{}-{}-' + best_classifier_name + '.pkl')

    # Displaying results
    s = '\nClassifier results {}-{}-{}'.format(class_name, feature, rel_pos)
    for c, score in scores.values():
        s += '\n   >>> {} : {}'.format(c.get_name(), score)
    s += '\nBest classifier: {} => {}'.format(best_classifier_name, best_score)
    s += '\nSaving at "{}"'.format(save_path)
    print(s)

    # Creates the necessary directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode="wb") as fp:
        pickle.dump(best_classifier.classifier, fp)

    if not do_plot:
        return

    # Plotting...
    y = label_binarize(y_test, classes=[0, 1])

    info = []
    for clf in classifiers:
        x, y, auc = clf.get_roc_curve(y, x_test)
        info.append((x, y, auc))

    if save_plot_info:
        # Saving to pickle file...
        save_path = mkd('trained_classifiers/plot-{}-{}-{}.pkl')

        with open(save_path, mode="wb") as fp:
            pickle.dump(info, fp)
    else:
        graph_path = mkd('trained_classifiers/{}-{}-{}.png')
        plot_classifier_roc(info, display, graph_path)


def plot_from_info(rel_pos, feature, image_class=None, display=True, **kwargs):
    class_name = image_class if image_class is not None else 'all'

    project_base = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../..'))

    def mkd(pattern):
        return os.path.join(project_base, pattern.format(class_name, feature, rel_pos))

    save_path = mkd('trained_classifiers/plot-{}-{}-{}.pkl')
    with open(save_path, mode="rb") as fp:
         info = pickle.load(fp)

    graph_path = mkd('trained_classifiers/{}-{}-{}.png')
    plot_classifier_roc(info, display, graph_path)


def plot_classifier_roc(info, display, graph_path):
    import matplotlib.pyplot as plt

    [
        (fpr_lregression, tpr_lregression, roc_auc_lregression),
        (fpr_linsvm, tpr_linsvm, roc_auc_linsvm),
        (fpr_knn, tpr_knn, roc_auc_knn),
        (fpr_clf, tpr_clf, roc_auc_clf),
        (fpr_rfc, tpr_rfc, roc_auc_rfc),
    ] = info

    plt.figure()
    plt.plot(fpr_lregression, tpr_lregression, color='cornflowerblue', label='ROC curve lregression (area = %0.2f)' % roc_auc_lregression)
    plt.plot(fpr_linsvm, tpr_linsvm, color='green', label='ROC curve LINSVM (area = %0.2f)' % roc_auc_linsvm)
    plt.plot(fpr_knn, tpr_knn, color='aqua', label='ROC curve KNN (area = %0.2f)' % roc_auc_knn)
    plt.plot(fpr_clf, tpr_clf, color='black', label='ROC curve clf (area = %0.2f)' % roc_auc_clf)
    plt.plot(fpr_rfc, tpr_rfc, color='darkgreen', label='ROC curve RFC (area = %0.2f)' % roc_auc_rfc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    if display:
        plt.show()
    else:
        plt.savefig(graph_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features', help='coma-separated list of features', type=str)
    parser.add_argument('-i', '--images', help='coma-separated list of image classes', type=str)
    args = parser.parse_args()

    # Features
    features = [VGG16FeatureExtractor.name(), L2FeatureExtractor.name(), ColourFeatureExtractor.name(),
                GradientFeatureExtractor.name()]
    if args.features is not None:
        feats_names = [f.strip() for f in args.features.split(',')]
        features = [f for f in feats_names if f in features]
    print('Using features: {}'.format(features))

    # Image classes
    images = ['animals', 'art', 'cities', 'landscapes', 'portraits', 'space', None]
    if args.images is not None:
        image_names = [i.strip() for i in args.images.split(',')]
        images = [i for i in image_names if i in images]
        if 'None' in image_names:
            images += [None]
    print('Using images: {}'.format(images))

    # Relative positions
    rel_positions = [p.value for p in RelativePosition]

    jobs = []
    for f in features:
        for c in images:
            for rel_pos in rel_positions:
                print('\nTraining classifier for {}-{}-{}...'.format(f, c, rel_pos))

                p = multiprocessing.Process(
                    target=train_classifiers,
                    args=(rel_pos, f, c),
                    kwargs=dict(do_plot=True, save_plot_info=True, display=False)
                )
                jobs.append(p)
                p.start()

    for p in jobs:
        p.join()
