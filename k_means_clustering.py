"""
k-means clustering.

Matthew Alger
2015
"""

import numpy

class KMeansClusterer(object):

    """
    Simple k-means clusterer.
    """

    def __init__(self, k, d, means=None):
        """
        k: Number of means. int.
        d: Dimensions in data. int.
        means: Initial mean positions (optional). [[float]].
        -> KMeansClusterer
        """

        self.k = k
        self.d = d

        if means:
            assert means.shape == (self.k, self.d)
            self.means = means
        else:
            self.means = numpy.random.random((self.k, self.d))

    def train(self, data, iterations):
        """
        Train the clusterer.

        data: Data to cluster. [[float]].
        iterations: How many training steps to run. int.
        """

        # Assign each data point to a cluster.
        cluster = numpy.random.randint(self.k, size=(data.shape[0],))

        # Find centroids.
        for c in range(self.k):
            data