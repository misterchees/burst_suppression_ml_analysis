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

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        if hasattr(self.model, "probability") and self.model.probability:
            return self.model.predict_proba(x)
        raise AttributeError("SVM was not initialized with probability=True.")
