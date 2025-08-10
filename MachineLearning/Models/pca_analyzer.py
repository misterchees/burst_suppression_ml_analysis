import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')


class PCAAnalyzer:
    def __init__(self, n_components=2):
        """
        Initializes PCA analyzer.

        :param n_components: Number of PCA components to keep.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.columns = None
        self.pca_result = None
        self.variance_ratio = None

    def fit_transform(self, df: pd.DataFrame):
        """
        Fits PCA to the feature data and returns the transformed result.

        :param df: Full feature DataFrame (with possible metadata columns).
        :returns: PCA-transformed numpy array.
        """
        df_clean = df.drop(columns=["Start", "End", "ResultID", "label"], errors="ignore")
        self.columns = df_clean.columns

        X_scaled = self.scaler.fit_transform(df_clean)
        self.pca_result = self.pca.fit_transform(X_scaled)
        self.variance_ratio = self.pca.explained_variance_ratio_

        return self.pca_result

    def plot_components_2d(self, labels=None, figsize=(8, 6),
                           jitter=False, jitter_strength=0.01,
                           alpha=0.6, marker_size=20,
                           separate_plots=False):
        """
        Visualizes the first two principal components from PCA results in a 2D scatter plot.
        Allows customization of the visualization and supports optional jittering of the points.
        If labels are provided, points will be color-coded accordingly. Optionally, separate plots
        can be created for each label.

        :param labels: Optional. Array-like object containing labels for the data points.
                       If provided, points will be color-coded based on these labels.
        :param figsize: Tuple specifying the width and height of the plot in inches. Defaults to (8, 6).
        :param jitter: Boolean flag indicating whether to add random noise (jitter) to the points. Default is False.
        :param jitter_strength: Float specifying the standard deviation of the noise applied if jitter is True. Default is 0.01.
        :param alpha: Float specifying the transparency level of the scatter plot points. Accepts values between 0 and 1.
                      Defaults to 0.6.
        :param marker_size: Integer specifying the size of the scatter plot markers. Defaults to 20.
        :param separate_plots: Boolean flag indicating whether separate plots should be created for each label group.
                               Defaults to False.
        :return: None
        """
        if self.pca_result is None:
            raise ValueError("Run fit_pca() first.")

        X = self.pca_result[:, :2]  # Erste 2 PCs

        if jitter:
            noise = np.random.normal(0, jitter_strength, X.shape)
            X = X + noise

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            if separate_plots:
                for ul in unique_labels:
                    mask = labels == ul
                    plt.figure(figsize=figsize)
                    plt.scatter(X[mask, 0], X[mask, 1],
                                alpha=alpha, s=marker_size, label=str(ul))
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.title(f"PCA Projection - Label {ul}")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
            else:
                plt.figure(figsize=figsize)
                for ul in unique_labels:
                    mask = labels == ul
                    plt.scatter(X[mask, 0], X[mask, 1],
                                alpha=alpha, s=marker_size, label=str(ul))
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title("PCA Projection")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        else:
            plt.figure(figsize=figsize)
            plt.scatter(X[:, 0], X[:, 1], alpha=alpha, s=marker_size)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA Projection")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_components_3d(self, labels=None, jitter=False, jitter_strength=0.01,
                           alpha=0.6, marker_size=20, separate_plots=False):
        """
        Generates a 3D scatter plot using the results of a PCA (Principal Component
        Analysis). It supports optional jittering of the points for better visibility,
        the inclusion of labels for grouping data points by categories, and the
        ability to create separate plots for each label group or consolidate all
        groups into a single plot.

        :param labels: Optional. An array-like object containing labels for the data
            points. The labels are used to group data points by categories. If None,
            no grouping is applied.
        :type labels: Optional[Iterable]
        :param jitter: Optional. A boolean that determines whether jitter should be
            added to the points in the plot. Jittering can help distinguish overlapping
            points. Defaults to False.
        :type jitter: bool
        :param jitter_strength: Optional. The standard deviation of the normal
            distribution used to add jitter to the points. Only used if `jitter` is
            set to True. Defaults to 0.01.
        :type jitter_strength: float
        :param alpha: Optional. The transparency level of the points in the plot.
            Should be between 0 and 1. Defaults to 0.6.
        :type alpha: float
        :param marker_size: Optional. The size of the markers representing the points
            in the plot. Defaults to 20.
        :type marker_size: int
        :param separate_plots: Optional. A boolean indicating whether separate plots
            should be created for each category specified in the labels. If False,
            all data is plotted in a single 3D scatter plot. Defaults to False.
        :type separate_plots: bool
        :return: None. The method displays the 3D scatter plot(s) but does not return
            any value.
        :rtype: None
        """
        if self.pca_result is None:
            raise ValueError("Run fit_transform() first.")

        X = self.pca_result[:, :3]  # erste 3 PCs

        if jitter:
            noise = np.random.normal(0, jitter_strength, X.shape)
            X = X + noise

        if labels is not None:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

            if separate_plots:
                # For every label a single plot
                for ul in unique_labels:
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    idx = labels == ul
                    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2],
                               label=str(ul), alpha=alpha, s=marker_size)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("PC3")
                    ax.set_title(f"PCA 3D Plot - Label {ul}")
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
            else:
                # Everything in one plot
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                for ul in unique_labels:
                    idx = labels == ul
                    ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2],
                               label=str(ul), alpha=alpha, s=marker_size)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_zlabel("PC3")
                ax.set_title("PCA 3D Plot")
                ax.legend()
                plt.tight_layout()
                plt.show()

        else:
            # If no labels given → Everything in one plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=alpha, s=marker_size)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("PCA 3D Plot")
            plt.tight_layout()
            plt.show()

    def plot_scree(self):
        """
        Plots the explained variance ratio per component (scree plot).
        """
        if self.variance_ratio is None:
            raise ValueError("Run fit_transform() first.")

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, len(self.variance_ratio)+1), self.variance_ratio, marker='o')
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_feature_contributions(self, pc_index=0, top_n=10):
        """
        Returns the top contributing features for a given principal component.

        :param pc_index: Index of the component (0 = PC1, 1 = PC2, ...)
        :param top_n: Number of top features to return.
        :returns: DataFrame with feature names and contribution weights.
        """
        if self.pca.components_ is None:
            raise ValueError("Run fit_transform() first.")

        component = self.pca.components_[pc_index]
        contributions = pd.Series(np.abs(component), index=self.columns)
        return contributions.sort_values(ascending=False).head(top_n)

