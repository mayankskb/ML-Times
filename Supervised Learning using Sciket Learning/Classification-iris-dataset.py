from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

if __name__ == '__main__':
    iris = datasets.load_iris()
    y = iris['target']
    X = iris['data']
    knn = KNeighborsClassifier(n_neighbors = 6)
    knn.fit(X, y)
    df = pd.read_csv('testdata.csv')
    predicted = knn.predict(X)
    for x in predicted:
        if x == 0:
            print("setosa")
        elif x == 1:
            print("versicolor")
        else:
            print("virginica")
            