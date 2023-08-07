import numpy as np


def compute_euclidean_distance(p1, p2):
    """
        This method computes the euclidean distance/sum of square error
    :param p1:
    :type p1:
    :param p2:
    :type p2:
    :return:
    :rtype:
    """
    # euclidean distance/sum of square error
    d = np.square(np.sum((p1 - p2) ** 2))
    return d


def compute_distance(p1, p2):
    """
        This method computes the distance between two data points
    :param p1:
    :type p1:
    :param p2:
    :type p2:
    :return:
    :rtype:
    """
    return np.sum(np.abs(p1 - p2))
