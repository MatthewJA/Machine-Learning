"""
Mixture models.

Matthew Alger
2015
"""

import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture(object):

    """
    Expectation-maximisation trained Gaussian mixture.
    """

    def __init__(self, k, d):
        """
        k: Number of Gaussians. int.
        d: Dimensions in data. int.
        -> GaussianMixture
        """

        self.k = k
        self.d = d
        self.means = np.zeros((self.k, self.d))
        self.covariances = [np.identity(self.d) for _ in range(self.k)]
        self.coefficients = np.ones(self.k)/self.k

    def train(self, data, iterations):
        """
        Train the mixture model.

        data: Data to train based on. [[float]] shape (n, d).
        iterations: How many training steps to run. int.
        """

        assert data.shape[1] == self.d

        self.initialise_means(data)

        self.gamma = np.ones((data.shape[0], self.k))/self.k
        for _ in range(iterations):
            # Calculate gammas.
            for i in range(data.shape[0]):
                for k in range(self.k):
                    # p(x given params)
                    px = (1/((2*np.pi)**(self.d/2) *
                             np.linalg.norm(self.covariances[k])**0.5) *
                          np.exp(-0.5*(data[i] - self.means[k]).T.dot(
                                       np.linalg.inv(self.covariances[k])).dot(
                                       data[i] - self.means[k])))
                    self.gamma[i, k] = px * self.coefficients[k]
            self.gamma /= np.sum(self.gamma, axis=1).reshape(self.gamma.shape[0], 1)

            # Calculate effective counts.
            self.effective_counts = np.sum(self.gamma, axis=0)

            # Calculate coefficients.
            self.coefficients = self.effective_counts/data.shape[0]

            # Calculate means.
            for k in range(self.k):
                self.means[k] = data.T.dot(self.gamma[:, k])/self.effective_counts[k]

            # Calculate covariances.
            for k in range(self.k):
                self.covariances[k] = sum(
                    self.gamma[n, k] * (data[0] - self.means[k]).reshape((self.d, 1)).dot((data[0] - self.means[k]).reshape((1, self.d)))
                    for n in range(data.shape[0]))/self.effective_counts[k]
            print(self.gamma)

    def initialise_means(self, data):
        """
        Initialises the means with the Forgy method.

        data: Data to initialise means for. [[float]] shape (n, d).
        """

        # Choose k random data points to be the initial means.
        points = np.random.randint(data.shape[0], size=(self.k,))
        self.means = data[points, :]


def test_gaussian_mixture():
    """
    Tests the Gaussian mixture on the Iris Plant Database.
    """

    from data import iris

    raw_data = iris.load_data(True, True)

    # Separate the classifications from the input data.
    data = raw_data[:, :-1]
    classifications = raw_data[:, -1]

    k = 3
    d = data.shape[1]

    gm = GaussianMixture(k, d)
    gm.train(data, 10)

    import matplotlib.pyplot as plt
    axes = 0, 3
    plt.plot(data[:, axes[0]], data[:, axes[1]], "kx")
    plt.plot(gm.means[:, axes[0]], gm.means[:, axes[1]], "bo")
    plt.show()
    print(gm.means)

if __name__ == '__main__':
    test_gaussian_mixture()
