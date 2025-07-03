"""This module contains the SVMGridSearchCV class."""


class SVMGridSearch:
    """This class implements a GridSearchCV class to find the best hyperparameters for a SVM classifier."""
    def __init__(self, svc_model, X, y, cv, scoring='accuracy'):
        """
        Initializes the SVMGridSearch instance.
        :param svc_model: The SVM model. Should be a base model.
        :param X: Unlabeled training data.
        :param y: Labels of the training data.
        :param cv: cross-validation generator. Accepts an iterable of splits (train, test)
        :param scoring: The metric that shall be optimized.
        """
        self.svc_model = svc_model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.grid = None

    def run(self, param_grid=None):
        """
        Initialize a GridSearchCV instance, fit for all hyperparameters und return it.
        :param param_grid: A dict containing the hyperparameters to tune.
        :return: The fitted GridSearchCV instance.
        """
        from sklearn.model_selection import GridSearchCV

        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': [0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'linear'],
            }

        self.grid = GridSearchCV(self.svc_model, param_grid, cv=self.cv, scoring=self.scoring, verbose=3)
        self.grid.fit(self.X, self.y)
        return self.grid

    def best_params(self):
        """Returns the hyperparameters of the best estimator."""
        return self.grid.best_params_ if self.grid else None

    def best_score(self):
        """Returns the mean of the cross-validated scores from the best estimator."""
        return self.grid.best_score_ if self.grid else None

    def best_estimator(self):
        """Returns the best estimator for the GridSearchCV instance. (highest score or smallest loss)"""
        return self.grid.best_estimator_ if self.grid else None
