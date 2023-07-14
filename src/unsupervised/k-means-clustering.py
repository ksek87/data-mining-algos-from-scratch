"""
K-Means Clustering Algorithm

This Python script implements the K-Means Clustering algorithm for unsupervised data clustering.
It randomly initializes K centroids, assigns each data point to the nearest centroid, and updates centroids iteratively.
The algorithm aims to minimize the within-cluster sum of squares (euclidean distance error) and
partitions the data into K distinct clusters.
It has applications in customer segmentation, image compression, and anomaly detection.

Note: Performance depends on centroid placement and choice of K.
"""

import numpy as np
import utils.methods as ut


class KMeans:
    def __init__(self, n_clusters, n_iter, len_data) -> None:
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.cluster_centroids = [0] * self.n_clusters
        self.data = []
        self.len_data = len_data
        self.assigned_data = np.zeros(self.len_data)

    def fit(self, X):
        self.data = X
        self.assigned_data = np.zeros(self.len_data)  # track the cluster assignment by index for the data

        # initialize  random centroids
        nums = np.random.choice(self.len_data, self.n_clusters, replace=False)
        for i in range(self.n_clusters):
            num = nums[i]  # np.random.randint(self.len_data)
            self.cluster_centroids[i] = np.asarray(X[num])

        # N-iter loop
        for i in range(self.n_iter):
            # choose a centroid for each point
            for j in range(self.len_data):
                centroid_idx = self._compute_best_centroid(X[j])
                # put points in the cluster closest
                self.assigned_data[j] = centroid_idx

            new_cents = []
            for cent in range(len(self.cluster_centroids)):
                clustered_data = [X[i] for i in range(self.len_data) if self.assigned_data[i] == cent]
                # re-calculate the cluster centroids
                centroid = np.mean(clustered_data, axis=0)
                new_cents.append(centroid)
                # self.cluster_centroids[cent] = centroid

            # check for convergence
            if np.allclose(self.cluster_centroids, new_cents):
                print(i, 'iterations completed')
                break
            else:
                self.cluster_centroids = new_cents
                # self.cluster_centroids[cent] = centroid

        return self.assigned_data, self.cluster_centroids

    def _compute_best_centroid(self, point):
        # determine best centroid based on euclidean distance
        dists = []
        for i in range(self.n_clusters):
            dists.append(ut.compute_euclidean_distance(point, self.cluster_centroids[i]))

        centroid_selection = np.argmin(dists)
        return centroid_selection

    def predict(self, data):
        pred = self._compute_best_centroid(data)
        return pred
