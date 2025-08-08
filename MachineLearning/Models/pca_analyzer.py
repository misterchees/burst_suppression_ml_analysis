import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    def plot_components(self, labels=None, figsize=(8, 6)):
        """
        Plots the first two principal components in 2D.

        :param labels: Optional labels for color-coding (e.g., classes).
        :param figsize: Tuple for figure size.
        """
        if self.pca_result is None:
            raise ValueError("Run fit_transform() first.")

        plt.figure(figsize=figsize)
        if labels is not None:
            unique_labels = np.unique(labels)
            for ul in unique_labels:
                mask = (labels == ul)
                plt.scatter(self.pca_result[mask, 0], self.pca_result[mask, 1], label=str(ul), alpha=0.6)
            plt.legend()
        else:
            plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], alpha=0.6)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection")
        plt.grid(True)
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

