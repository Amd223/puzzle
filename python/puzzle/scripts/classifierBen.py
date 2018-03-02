import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

dataset_file = "dataset.pkl"
with open(dataset_file, mode="rb") as fp:
    X_train, y_train = pickle.load(fp)

dataset_file_test = "datatest.pkl"
with open(dataset_file_test, mode="rb") as fp:
    X_test, y_test = pickle.load(fp)

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Grid of parameter values to search over
tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-5, 1e-6],
                     'C': [1, 100, 10000]}]
                   # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

tuned_parameters_rfc = [{'n_estimators':[1,2,5,10,100]}]


for classifer, params in [(RandomForestClassifier, tuned_parameters_rfc), (SVC, tuned_parameters_svm)]:

    score = "accuracy"

    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(classifer(), params, cv=3, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
