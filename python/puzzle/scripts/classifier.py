import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

dataset_file = "dataset_down.pkl"
with open(dataset_file, mode="rb") as fp:
    X_train, y_train = pickle.load(fp)

dataset_file_test = "datatest_down.pkl"
with open(dataset_file_test, mode="rb") as fp:
    X_test, y_test = pickle.load(fp)

#print(X_train.shape, X_test.shape)

# print(len(y_test))
#
# #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)


lregression = LogisticRegression()
lregression.fit(X_train, y_train)
print(lregression.score(X_test, y_test))

svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

linsvm = LinearSVC()
linsvm.fit(X_train, y_train)
print(linsvm.score(X_test, y_test))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))
#y_score = rfc.fit(X_train, y_train).decision_function(X_test)
#rfc.roc_curve(y_test,y_score )

dataset_file = "rfc.pkl"
with open(dataset_file, mode="wb") as fp:
    pickle.dump(rfc, fp)

y = label_binarize(y_test, classes=[0, 1])

pred_svm = svm.decision_function(X_test)
fpr_svm, tpr_svm, thresholds = roc_curve(y, pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

pred_linsvm = linsvm.decision_function(X_test)
fpr_linsvm, tpr_linsvm, thresholds = roc_curve(y, pred_linsvm)
roc_auc_linsvm = auc(fpr_linsvm, tpr_linsvm)

pred_knn = knn.predict(X_test)
fpr_knn, tpr_knn, thresholds = roc_curve(y, pred_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

pred_lregression = lregression.decision_function(X_test)
fpr_lregression, tpr_lregression, thresholds = roc_curve(y, pred_lregression)
roc_auc_lregression = auc(fpr_lregression, tpr_lregression)

pred_clf = clf.predict(X_test)
fpr_clf, tpr_clf, thresholds = roc_curve(y, pred_clf)
roc_auc_clf = auc(fpr_clf, tpr_clf)

pred_rfc = rfc.predict(X_test)
fpr_rfc, tpr_rfc, thresholds = roc_curve(y, pred_rfc)
roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
# obtenir les scores
# 2 variables en output?

plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange', label='ROC curve SVM (area = %0.2f)' % roc_auc_svm)
plt.plot(fpr_linsvm, tpr_linsvm, color='green', label='ROC curve LINSVM (area = %0.2f)' % roc_auc_linsvm)
plt.plot(fpr_knn, tpr_knn, color='aqua', label='ROC curve KNN (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_lregression, tpr_lregression, color='cornflowerblue', label='ROC curve lregression (area = %0.2f)' % roc_auc_lregression)
plt.plot(fpr_clf, tpr_clf, color='black', label='ROC curve clf (area = %0.2f)' % roc_auc_clf)
plt.plot(fpr_rfc, tpr_rfc, color='darkgreen', label='ROC curve RFC (area = %0.2f)' % roc_auc_rfc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()