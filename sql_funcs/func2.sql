CREATE OR REPLACE FUNCTION train_data() 
    RETURNS float
AS $$
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    dataset1 = datasets.load_iris()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1.data, dataset1.target, test_size=0.2)

    from diffprivlib.models import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train1, y_train1)

    clf.predict(X_test1)
    score = clf.score(X_test1, y_test1)

    return score
$$
LANGUAGE 'plpython3u';