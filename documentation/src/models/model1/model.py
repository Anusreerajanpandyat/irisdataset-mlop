from sklearn.svm import SVC

class SVMClassifier:
    """
    Support Vector Machine classifier for the Iris dataset.
    """
    def __init__(self, kernel='linear', C=1.0):
        self.clf = SVC(kernel=kernel, C=C, random_state=42)

    def train(self, X_train_std, y_train):
        self.clf.fit(X_train_std, y_train)

    def predict(self, X_test_std):
        y_pred = self.clf.predict(X_test_std)
        return y_pred
