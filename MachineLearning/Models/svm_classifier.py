from sklearn.svm import SVC


class SVMClassifier:
    """
    Wrapper class for SVM classifier using sklearn's SVC.
    """

    def __init__(self, **svm_kwargs):
        """
        :param svm_kwargs: Any parameters passed to sklearn.svm.SVC
        """
        self.model = SVC(**svm_kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "probability") and self.model.probability:
            return self.model.predict_proba(X)
        raise AttributeError("SVM was not initialized with probability=True.")
