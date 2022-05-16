CREATE OR REPLACE FUNCTION train_data () 
    RETURNS float
AS $$
    import psycopg2

    import pandas as pd
    import numpy as np
$$
LANGUAGE 'plpython3u';


CREATE EXTENSION plpythonu;
CREATE EXTENSION plpython3u;







CREATE OR REPLACE FUNCTION train_data() 
    RETURNS float
AS $$
    import numpy as np
    import pandas as pd
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from diffprivlib.models import GaussianNB
    
    dataset = pd.DataFrame.from_records(plpy.execute('SELECT * FROM iris'))[['sepal_l','sepal_w','petal_l','petal_w','class']]

    iris_array = dataset[['sepal_l','sepal_w','petal_l','petal_w']].to_numpy()
    iris_array_target = dataset['class'].to_numpy()

    iris_array_target = np.where(iris_array_target == 'Iris-setosa', 0, iris_array_target)
    iris_array_target = np.where(iris_array_target == 'Iris-versicolor', 1, iris_array_target)
    iris_array_target = np.where(iris_array_target == 'Iris-virginica', 2, iris_array_target)
    
    X_train, X_test, y_train, y_test = train_test_split(iris_array, iris_array_target, test_size=0.2)

    dataset1 = datasets.load_iris()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1.data, dataset1.target, test_size=0.2)

    clf = GaussianNB()
    clf.fit(X_train1, y_train1)

    clf.predict(X_test1)
    score = clf.score(X_test1, y_test1)

    return score
$$
LANGUAGE 'plpython3u';


CREATE EXTENSION plpythonu;
CREATE EXTENSION plpython3u;