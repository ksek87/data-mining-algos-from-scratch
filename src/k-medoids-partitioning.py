"""
K-Medoids Partitioning Algorithm

The K-Medoids Partitioning algorithm, a clustering technique that partitions data into
K distinct clusters. Unlike K-Means, which uses centroids as representatives, K-Medoids selects actual data points (
medoids) as cluster representatives based on a distance metric. Then each medoid is evaluated for dissimilarity,
with the objective of minimizing dissimilarity through swapping medoid with non-medoids data points.

K-Medoids Partitioning is useful when dealing with outliers or when the concept of a centroid is ill-defined.
It offers robustness to noise and enhanced interpretability.
However, it can be computationally expensive due to pairwise dissimilarity calculations.

A further improvement to the algorithm is the PAM (Partitioning Around Medoids) Algorithm
The PAM algorithm provides a solution to  efficient medoid selection in clustering.
It iteratively updates medoids by minimizing the total distance of swapping them with non-medoid data points.
"""

import numpy as np
import utils as ut


class KMedoids:
    def __init__(self, n_clusters, n_iter, len_data):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.medoids = []
        self.data = []
        self.len_data = len_data
        self.assigned_data = np.zeros(self.len_data)

    def fit(self, X):
        self.data = X
        self.assigned_data = np.zeros(self.len_data)  # Track the cluster assignment by index for the data

        # Initialize random medoids
        medoid_indices = np.random.choice(self.len_data, self.n_clusters, replace=False)
        self.medoids = [X[i] for i in medoid_indices]

        # N-iter loop
        for _ in range(self.n_iter):
            # Assign each data point to the nearest medoid
            for j in range(self.len_data):
                medoid_idx = self._compute_best_medoid(X[j])
                self.assigned_data[j] = medoid_idx

            # Update medoids
            for i in range(self.n_clusters):
                cluster_data = [X[j] for j in range(self.len_data) if self.assigned_data[j] == i]
                cluster_size = len(cluster_data)
                total_distance = np.zeros(cluster_size)
                for j in range(cluster_size):
                    for k in range(cluster_size):
                        total_distance[j] += ut.compute_distance(cluster_data[j], cluster_data[k])

                best_medoid_idx = np.argmin(total_distance)
                self.medoids[i] = cluster_data[best_medoid_idx]

        return self.assigned_data, self.medoids

    def _compute_best_medoid(self, point):
        # Determine the best medoid based on distance
        distances = [ut.compute_distance(point, medoid) for medoid in self.medoids]
        best_medoid_idx = np.argmin(distances)
        return best_medoid_idx

    def predict(self, data):
        pred = self._compute_best_medoid(data)
        return pred


# TODO: Implement PAM algorithm + explain how this improves from k-medoids
class PAM:
    def __init__(self, n_clusters, n_iter, len_data):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.medoids = []
        self.data = []
        self.len_data = len_data
        self.assigned_data = np.zeros(self.len_data)

    def fit(self, X):
        self.data = X
        self.assigned_data = np.zeros(self.len_data)
        return self.assigned_data, self.medoids
