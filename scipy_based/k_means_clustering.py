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

    def __init__(self, k, d):
        """
        k: Number of means. int.
        d: Dimensions in data. int.
        -> KMeansClusterer
        """

        self.k = k
        self.d = d
        self.means = numpy.zeros((self.k, self.d))

    def train(self, data, iterations):
        """
        Train the clusterer.

        data: Data to cluster. [[float]] shape (n, d).
        iterations: How many training steps to run. int.
        """

        assert data.shape[1] == self.d

        self.initialise_means(data)

        for _ in range(iterations):
            # Assign each data point a cluster.
            clusters = self.classify(data)

            # Update the means to be the centroids of each cluster.
            for c in range(self.k):
                cluster_data = data[clusters == c]
                if cluster_data.shape[0]:
                    self.means[c] = cluster_data.mean(axis=0)

    def initialise_means(self, data):
        """
        Initialises the cluster means with the Forgy method.

        data: Data to initialise means for. [[float]] shape (n, d).
        """

        # Choose k random data points to be the initial means.
        points = numpy.random.randint(data.shape[0], size=(self.k,))
        self.means = data[points, :]

    def classify(self, data):
        """
        Classify data points.

        data: Data to classify. [[float]] shape (n, d).
        -> Classifications. [int] shape (n,).
        """

        clusters = numpy.zeros((data.shape[0],), dtype=int)
        for index, point in enumerate(data):
            cluster_distances = [numpy.linalg.norm(self.means[c] - point)
                                 for c in range(self.k)]
            clusters[index] = numpy.argmin(cluster_distances)

        return clusters


def test_clusterer():
    """
    Tests the clusterer on the Iris Plant Database.
    """

    from data import iris

    raw_data = iris.load_data(True, True)

    # Separate the classifications from the input data.
    data = raw_data[:, :-1]
    classifications = raw_data[:, -1]

    k = 3
    d = data.shape[1]

    clusterer = KMeansClusterer(k, d)
    clusterer.initialise_means(data)
    clusterer.train(data, 100)

    clusters = clusterer.classify(data)

    c0 = clusters == 0
    c1 = clusters == 1
    c2 = clusters == 2

    d0 = classifications == 0
    d1 = classifications == 0.5
    d2 = classifications == 1

    success = max((d0 == c0).mean(), (d1 == c0).mean(), (d2 == c0).mean())
    print("{:.0%}".format(success))

    import matplotlib.pyplot as plt
    plt.plot(data[clusters == 0][:, 0],
             data[clusters == 0][:, 3], "r.")
    plt.plot(data[clusters == 1][:, 0],
             data[clusters == 1][:, 3], "g.")
    plt.plot(data[clusters == 2][:, 0],
             data[clusters == 2][:, 3], "b.")
    plt.plot(clusterer.means[:, 0],
             clusterer.means[:, 3], "o", color="black")
    plt.show()

if __name__ == '__main__':
    test_clusterer()
