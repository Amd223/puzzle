import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

dataset_file = "dataset.pkl"
with open(dataset_file, mode="rb") as fp:
    X,Y = pickle.load(fp)

print(len(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

lregression = LogisticRegression()
lregression.fit(X_train, y_train)
print(lregression.score(X_test, y_test))

svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))