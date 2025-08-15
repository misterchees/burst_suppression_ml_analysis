import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from MachineLearning.Utils.plots import Plots
from MachineLearning.IO.save_result import SaveResult


class PCAAnalyzer:
    def __init__(self, hyperparams:dict, n_components=2):
        """
        Initializes PCA analyzer.

        :param n_components: Number of PCA components to keep.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.hyperparams = hyperparams
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

    def plot_components_2d(self, labels=None, figsize=(8, 6), jitter=False, jitter_strength=0.01, alpha=0.6,
                           marker_size=20, separate_plots=False, title="2D PCA Scatterplot", save_plot=False):
        """
        Visualizes the first two principal components from PCA results in a 2D scatter plot. For more details
        look into MachineLearning.Utils.plots.py
        """
        if self.pca_result is None:
            raise ValueError("Run fit_pca() first.")

        figs_and_axes = Plots.plot_components_2d(
            pca_result=self.pca_result,
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
            saver = SaveResult()
            if separate_plots:
                counter = 0
                for fig_and_ax in figs_and_axes:
                    fig, ax = fig_and_ax
                    saver.save_further_analysis(self.hyperparams, fig, "plot", "pca",
                                                title, f"part_{counter}")
                    counter += 1
            else:
                fig, ax = figs_and_axes
                saver.save_further_analysis(self.hyperparams, fig, "plot", "pca",
                                            title, "all_labels")



    def plot_components_3d(self, labels=None, jitter=False, jitter_strength=0.01, alpha=0.6, marker_size=20,
                           separate_plots=False, title="3D PCA Scatterplot", save_plot=False):
        """
        Visualizes the first three principal components from PCA results in a 3D scatter plot. For more details
        look into MachineLearning.Utils.plots.py
        """
        if self.pca_result is None:
            raise ValueError("Run fit_transform() first.")

        figs_and_axes = Plots.plot_components_3d(
            labels=labels,
            pca_result=self.pca_result,
            jitter=jitter,
            jitter_strength=jitter_strength,
            alpha=alpha,
            marker_size=marker_size,
            separate_plots=separate_plots,
            title=title
        )
        # Save figure after plotting it
        if save_plot:
            # Save figure after plotting it
            if save_plot:
                saver = SaveResult()
                if separate_plots:
                    counter = 0
                    for fig_and_ax in figs_and_axes:
                        fig, ax = fig_and_ax
                        saver.save_further_analysis(self.hyperparams, fig, "plot", "pca",
                                                    title, f"part_{counter}")
                        counter += 1
                else:
                    fig, ax = figs_and_axes
                    saver.save_further_analysis(self.hyperparams, fig, "plot", "pca",
                                                title, "all_labels")

    def plot_scree(self, save_plot=False):
        """
        Plots the explained variance ratio per component (scree plot).
        """
        if self.variance_ratio is None:
            raise ValueError("Run fit_transform() first.")

        fig, ax = Plots.plot_scree(self.variance_ratio)
        if save_plot:
            saver = SaveResult()
            saver.save_further_analysis(self.hyperparams, fig, "plot", "pca", "PCA_scree", "plot")

    def get_feature_contributions(self, pc_index=0, top_n=10, save_results=False):
        """
        Returns the top contributing features for a given principal component.

        :param pc_index: Index of the component (0 = PC1, 1 = PC2, ...)
        :param top_n: Number of top features to return.
        :param save_results: If True, saves the results to the further_analysis folder.
        :returns: Series with feature names and contribution weights.
        """
        if self.pca.components_ is None:
            raise ValueError("Run fit_transform() first.")

        component = self.pca.components_[pc_index]
        contributions = pd.Series(np.abs(component), index=self.columns)
        contributions.sort_values(ascending=False).head(top_n)
        if save_results:
            from MachineLearning.IO.save_result import SaveResult
            saver = SaveResult()
            saver.save_further_analysis(self.hyperparams, pd.DataFrame(contributions),"dataframe",
                                        "pca", f"PCA_top_{top_n}", f"feature_contributions_PC_{pc_index}")
        return contributions

    def get_points_in_region(self, labels, cluster_label, dims=2, confidence=0.95):
        """
        Returns indices of points in a cluster that lie within the confidence region
        (ellipse for dims=2, ellipsoid for dims=3) based on Mahalanobis distance.

        :param labels: Array-like cluster labels for each point in PCA space.
        :param cluster_label: The specific cluster to analyze.
        :param dims: Number of PCA dimensions to consider (2 or 3).
        :param confidence: Confidence level for the region (default=0.95).
        :returns: Indices of points inside the region.
        """
        from scipy.stats import chi2
        if self.pca_result is None:
            raise ValueError("PCA must be run first with fit_transform().")

        # Select the cluster points
        cluster_points = self.pca_result[np.array(labels) == cluster_label, :dims]

        # Compute center of mass
        center = cluster_points.mean(axis=0)

        # Covariance and inverse
        cov_matrix = np.cov(cluster_points, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Mahalanobis distances
        diffs = cluster_points - center
        dists_sq = np.sum(diffs @ inv_cov_matrix * diffs, axis=1)

        # Chi-squared threshold for given dimensions and confidence
        threshold = chi2.ppf(confidence, df=dims)

        # Indices of points inside the region
        cluster_indices = np.where(np.array(labels) == cluster_label)[0]
        inside_indices = cluster_indices[dists_sq <= threshold]

        return inside_indices

    def plot_cluster_with_ellipse(self, pca_data, labels, cluster_id, scale_factor=1.0, ax=None):
        """
        Plots cluster points and ellipse, returns indices inside ellipse.
        """
        if ax is None:
            fig, ax = plt.subplots()

        cluster_mask = labels == cluster_id
        ax.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1], label=f"Cluster {cluster_id}")

        indices = self.get_points_in_ellipse(pca_data, cluster_mask, scale_factor)

        # Draw ellipse
        points = pca_data[cluster_mask]
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * scale_factor * np.sqrt(eigvals)

        ellipse = Ellipse(mean, width, height, angle=angle, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(ellipse)

        ax.legend()
        return indices