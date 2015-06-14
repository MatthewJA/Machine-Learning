"""
Bayesian regression.

Matthew Alger
2015
"""

import numpy as np
import matplotlib.pyplot as plt


class BayesianRegression1D(object):

    """
    Implements one-dimensional Bayesian regression.
    """

    def __init__(self, basis_dimension):
        self.basis_dimension = basis_dimension
        self.prior = (np.zeros(self.basis_dimension),
                      np.zeros((self.basis_dimension, self.basis_dimension)))
        self.beta = 0
        self.n = 0
        self.average_target = 0
        self.average_target_sq = 0

    def train(self, feature_matrix, targets):
        # Update beta by first updating the running averages.
        self.average_target = (self.average_target*self.n +
                targets.sum()*targets.shape[0])/(self.n + targets.shape[0])
        self.average_target_sq = (self.average_target_sq*self.n +
                (targets**2).sum()*targets.shape[0])/(self.n + targets.shape[0])
        self.beta = 1/(self.average_target_sq - self.average_target**2)
        self.n += 1

        # Calculate the posterior distribution.
        inv_covariance_matrix = self.prior[1] + (self.beta *
            feature_matrix.T.dot(feature_matrix))
        mean = np.linalg.inv(inv_covariance_matrix).dot(
            self.prior[1].dot(self.prior[0]) + self.beta*feature_matrix.T.dot(
                targets))
        posterior = (mean, inv_covariance_matrix)

        # Update the prior.
        self.prior = posterior

def plot_gaussian(mean, inv_cov, dims):
    gaussian_data = np.random.multivariate_normal(br.prior[0],
        np.linalg.inv(br.prior[1]),
        size=(100000,))
    gaussian_histogram = np.histogram2d(gaussian_data[:,dims[0]],
        gaussian_data[:,dims[1]],
        bins=100)
    gaussian_heatmap = plt.pcolormesh(gaussian_histogram[0])
    plt.draw()

def make_data(N, M):
    r = 10
    xs = np.linspace(0, r, N)
    ys = np.sin(xs-r//2) + np.random.normal(0, size=(N,))
    xs = np.reshape(xs, xs.shape + (1,))
    feature_matrix = np.concatenate([xs**m for m in range(0, M)], axis=1)
    return xs, feature_matrix, ys

if __name__ == '__main__':
    N = 500
    M = 10
    br = BayesianRegression1D(M)
    plt.ion()
    for i in range(100):
        xs, feature_matrix, targets = make_data(N, M)
        br.train(feature_matrix, targets)
        # plot_gaussian(br.prior[0], br.prior[1], (1, 2))
        plt.cla()
        plt.plot(xs, feature_matrix.dot(br.prior[0]), "ro")
        plt.plot(xs, targets, "k+")
        plt.draw()
