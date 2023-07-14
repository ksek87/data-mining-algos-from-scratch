import numpy as np


def compute_euclidean_distance(p1, p2):
    # euclidean distance/sum of square error
    d = np.square(np.sum((p1 - p2) ** 2))
    return d


def compute_distance(p1, p2):
    # Compute the distance between two data points
    return np.sum(np.abs(p1 - p2))
