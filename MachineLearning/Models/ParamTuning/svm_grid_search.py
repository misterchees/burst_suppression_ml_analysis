class SVMGridSearch:
    def __init__(self, svc_model, X, y, cv=5, scoring='accuracy'):
        self.svc_model = svc_model
        self.X = X
        self.y = y
        self.cv = cv
        self.scoring = scoring
        self.grid = None

    def run(self, param_grid=None):
        from sklearn.model_selection import GridSearchCV

        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': [0.01, 0.1, 1],
                'kernel': ['rbf']
            }

        self.grid = GridSearchCV(self.svc_model, param_grid, cv=self.cv, scoring=self.scoring)
        self.grid.fit(self.X, self.y)
        return self.grid

    def best_params(self):
        return self.grid.best_params_ if self.grid else None

    def best_score(self):
        return self.grid.best_score_ if self.grid else None

    def best_estimator(self):
        return self.grid.best_estimator_ if self.grid else None
