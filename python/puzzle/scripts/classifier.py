import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

dataset_file = "dataset.pkl"
with open(dataset_file, mode="rb") as fp:
    X_train, y_train = pickle.load(fp)

dataset_file_test = "datatest.pkl"
with open(dataset_file_test, mode="rb") as fp:
    X_test, y_test = pickle.load(fp)

print(len(y_test))

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

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

classifier = rfc
classifier.fit(X_train, y_train)
y_score = classifier.predict_proba(X_test)