from sklearn.cluster import KMeans as SKKMeans
from MachineLearning.IO.save_result import SaveResult
from MachineLearning.Utils.path_manager import PathManager


class KMeans:
    """
    Wrapper for scikit-learn's KMeans implementation.

    :param n_clusters: Number of clusters (k).
    :param max_iter: Maximum number of iterations for convergence.
    :param tol: Convergence tolerance.
    :param random_state: Random seed for reproducibility.
    """
    def __init__(self, pm: PathManager, hyperparams: dict, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.model = SKKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            verbose=2
        )
        self.hyperparams = hyperparams
        self.labels_ = None
        self.centroids = None

        # Initialize IO Utilities
        self.pm = pm
        self.saver = SaveResult(self.pm)

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

    def plot_components_2d(self, data, labels=None, figsize=(8, 6), jitter=False, jitter_strength=0.01, alpha=0.6,
                           marker_size=20, separate_plots=False, title="2D PCA Scatterplot", save_plot=False):
        """
        Visualizes the first two principal components from PCA results in a 2D scatter plot. For more details
        look into MachineLearning.Utils.plots.py
        """
        from MachineLearning.Models.pca_analyzer import PCAAnalyzer
        from MachineLearning.Utils.plots import Plots
        analyzer = PCAAnalyzer(self.pm, self.hyperparams, n_components=2)
        # reduce data - result is stored internally of this class
        pca_result = analyzer.fit_transform(data)

        figs_and_axes = Plots.plot_components_2d(
            pca_result=pca_result,
            title=title,
            labels=labels,
            figsize=figsize,
            jitter=jitter,
            jitter_strength=jitter_strength,
            alpha=alpha,
            marker_size=marker_size,
            separate_plots=separate_plots
        )
        # Save figure after plotting it
        if save_plot:
            self.saver.save_plots(self.hyperparams, "k_means", figs_and_axes, title)



    def plot_components_3d(self, data, labels=None, jitter=False, jitter_strength=0.01, alpha=0.6, marker_size=20,
                           separate_plots=False, title="3D PCA Scatterplot", save_plot=False):
        """
        Visualizes the first three principal components from PCA results in a 3D scatter plot. For more details
        look into MachineLearning.Utils.plots.py
        """
        from MachineLearning.Models.pca_analyzer import PCAAnalyzer
        from MachineLearning.Utils.plots import Plots
        analyzer = PCAAnalyzer(self.pm, self.hyperparams, n_components=3)
        # reduce data - result is stored internally of this class
        pca_result = analyzer.fit_transform(data)

        figs_and_axes = Plots.plot_components_3d(
            labels=labels,
            pca_result=pca_result,
            jitter=jitter,
            jitter_strength=jitter_strength,
            alpha=alpha,
            marker_size=marker_size,
            separate_plots=separate_plots,
            title=title
        )
        # Save figure after plotting it
        if save_plot:
            self.saver.save_plots(self.hyperparams, "k_means", figs_and_axes, title)
