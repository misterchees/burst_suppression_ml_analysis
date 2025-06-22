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

    def train(self, x_train, y_train):
        """
        Trains the model with given training data.
        :param x_train: Training data
        """
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        if hasattr(self.model, "probability") and self.model.probability:
            return self.model.predict_proba(x)
        raise AttributeError("SVM was not initialized with probability=True.")

    def tune_hyperparameters(self, X, y, scoring='accuracy'):

        tuner = SVMGridSearch(self.get_base_model(), X, y, scoring=scoring)
        tuner.run()
        self.model = tuner.best_estimator()

    @staticmethod
    def get_base_model():
        return SVC()  # oder mit default self-params
