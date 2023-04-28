

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from splitting import split_data


def test_split_data():
    iris = load_iris(as_frame=True)
    data = iris.frame
    X_train, X_test, y_train, y_test = split_data(data)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
