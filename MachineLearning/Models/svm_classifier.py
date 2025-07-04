"""This module implements a class for the SVM classifier from sklearn."""
from sklearn.svm import SVC
from MachineLearning.Models.ParamTuning.svm_grid_search import SVMGridSearch


class SVMClassifier:
    """
    Wrapper class for SVM classifier using sklearn's SVC.
    """

    def __init__(self, **svm_kwargs):
        """
        Initializes SVM classifier with given parameters.
        :param svm_kwargs: Any parameters passed to sklearn.svm.SVC
        """
        self.model = SVC(**svm_kwargs)

    def train(self, X_train, y_train):
        """
        Trains the model with given training data.

        :param X_train: Unlabeled training data
        :param y_train: Labels for given training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Performs classification on given data.

        :param X: Unlabeled test data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts probabilities for possible outcomes. Model needs to have probability information
        computed at training time.
        :param X: unlabeled test data.
        """
        if hasattr(self.model, "probability") and self.model.probability:
            return self.model.predict_proba(X)
        raise AttributeError("SVM was not initialized with probability=True.")

    def tune_hyperparameters(self, X, y, cv, scoring='accuracy'):
        """
        Performs hyperparameter tuning on given Data for given scoring.
        :param X:
        :param y:
        :param cv:
        :param scoring:
        :return:
        """

        tuner = SVMGridSearch(self.get_base_model(), X, y, cv, scoring=scoring)
        tuner.run()
        self.model = tuner.best_estimator()

    @staticmethod
    def get_base_model():
        return SVC()  # oder mit default self-params
