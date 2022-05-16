import psycopg2
import pandas as pd
import numpy as np
from config import config

params = config()
conn = psycopg2.connect(**params)
cur = conn.cursor()

def create_pandas_table(sql_query, database = conn):
    table = pd.read_sql_query(sql_query, database)
    return table

dataset = create_pandas_table("SELECT * FROM iris")

cur.close()
conn.close()

iris_array = dataset[['sepal_l','sepal_w','petal_l','petal_w']].to_numpy()
iris_array_target = dataset['class'].to_numpy()

iris_array_target = np.where(iris_array_target == 'Iris-setosa', 0, iris_array_target)
iris_array_target = np.where(iris_array_target == 'Iris-versicolor', 1, iris_array_target)
iris_array_target = np.where(iris_array_target == 'Iris-virginica', 2, iris_array_target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_array, iris_array_target, test_size=0.2)

from sklearn import datasets
dataset1 = datasets.load_iris()
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1.data, dataset1.target, test_size=0.2)

from diffprivlib.models import GaussianNB
clf = GaussianNB()
clf.fit(X_train1, y_train1)

clf.predict(X_test1)
score = clf.score(X_test1, y_test1)
#print("Test accuracy: %f" % score)