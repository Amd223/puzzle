import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA

dataset_file = "dataset.pkl"
with open(dataset_file, mode="rb") as fp:
    X_train, y_train = pickle.load(fp)

dataset_file_test = "datatest.pkl"
with open(dataset_file_test, mode="rb") as fp:
    X_test, y_test = pickle.load(fp)

print(len(y_train))

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

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