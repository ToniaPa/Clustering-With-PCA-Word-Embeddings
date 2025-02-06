"""
Module for implementing Agglomerative clustering algorithm
(copied from teacher)
"""

from sklearn.cluster import AgglomerativeClustering


class AgglomerativeClusteringAlgorithm:
    """
    Agglomerative clustering implementation.

    Attributes:
        n_clusters (int): Number of clusters to find.
        affinity (str): Metric used to compute linkage.
        linkage (str): Linkage criteria ('ward', 'complete', 'average', 'single').
    """

    def __init__(self, n_clusters=5, metric='euclidean', linkage='ward'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.metric, linkage=self.linkage)

    def fit_predict(self, X):
        """
        Fit the model and predict clusters.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)
