from sklearn.cluster import KMeans as SKKMeans


class KMeans:
    """
    Wrapper for scikit-learn's KMeans implementation.

    :param n_clusters: Number of clusters (k).
    :param max_iter: Maximum number of iterations for convergence.
    :param tol: Convergence tolerance.
    :param random_state: Random seed for reproducibility.
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.model = SKKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
        self.labels_ = None
        self.centroids = None

    def fit(self, X):
        """
        Fit the K-Means model to the given data.

        :param X: Data array of shape (n_samples, n_features).
        :returns: self
        """
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.centroids = self.model.cluster_centers_
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: Data array of shape (n_samples, n_features).
        :returns: Array of cluster labels.
        """
        return self.model.predict(X)

    def fit_predict(self, X):
        """
        Fit the model and return the cluster labels.

        :param X: Data array of shape (n_samples, n_features).
        :returns: Array of cluster labels.
        """
        self.fit(X)
        return self.labels_
