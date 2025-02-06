"""
Module for implementing HDBSCAN clustering algorithm
(copied from teacher)
"""

import hdbscan

class HDBSCANClustering:
    """
    HDBSCAN clustering implementation.

    Attributes:
        min_cluster_size (int): The minimum size of clusters.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
        metric (str): The metric to use when calculating distance between instances in a feature array.
    """

    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean'):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples,
                                     metric=self.metric)

    def fit_predict(self, X):
        """
        Fit the model and predict clusters.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)