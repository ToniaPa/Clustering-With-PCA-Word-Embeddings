"""
Module for implementing KMeans clustering algorithm
(copied from teacher)
"""

from sklearn.cluster import KMeans

class KMeansClustering:
    """
    KMeans clustering implementation.

    Attributes:
        n_clusters (int): Number of clusters to form.
        init (str): Method for initialization.
        n_init (int): Number of time the k-means algorithm will run with different centroid seeds.
        max_iter (int): Maximum number of iterations of the k-means algorithm.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
    """

    def __init__(self, n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
                            max_iter=self.max_iter, random_state=self.random_state)

    def fit_predict(self, X):
        """
        Fit the model and predict clusters.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.model.fit_predict(X)

    def get_inertia(self):
        """
          Function for returning inertia.
          Returns:
             inertia_ (Any): Sum of squared distances of samples to their closest cluster center,
                            weighted by the sample weights if provided.
         """
        return self.model.inertia_

    def get_centers(self):
        """
          Function for returning cluster centers.
          Returns:
             cluster_centers_ (Any): Coordinates of cluster centers.
         """
        return self.model.cluster_centers_

    def get_labels(self):
        """
          Function for returning labels.
          Returns:
             labels_ (Any): Labels of each point
         """
        return self.model.labels_
