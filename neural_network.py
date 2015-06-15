"""
Implements a simple neural network.

Matthew Alger
2015
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

class NeuralNetwork(object):

    """
    A simple neural network.
    """

    def __init__(self, structure, learning_rate, momentum):
        """
        structure: Tuple of dimensions of layers, with structure[0] being the
            input dimension and structure[-1] being the output dimension.
        learning_rate: How fast the model learns.
        momentum: How much the model weights its past experiences.
        """

        assert len(structure) >= 2
        assert 0 <= learning_rate <= 1
        assert 0 <= momentum <= 1

        self.structure = structure
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = len(self.structure)
        self.weights = [
            np.random.random((self.structure[l], self.structure[l+1]))
            for l in range(self.layers-1)]
        self.biases = [
            np.zeros((1, self.structure[l+1]))
            for l in range(self.layers-1)]

    def train(self, data, targets, iterations):
        """
        Train the weights in this neural network.
        data: (n, d) matrix, where n is the number of data points and d is the
            input dimension.
        targets: (n, d) matrix, where d is the output dimension.
        """

        assert data.shape[1] == structure[0]
        assert targets.shape[0] == data.shape[0]
        assert targets.shape[1] == structure[-1]

        for iteration in range(iterations):
            # Forward propagation...
            activations = self.activations(data) # outputs from layer 1 .. l
                                                 # (n, dl)

            # Back propagation...
            deltas = self.deltas(activations, targets) # deltas from layer 1 .. l
                                                       # (n, dl)

            # Gradient descent!
            for l in range(self.layers-1):
                for n in range(data.shape[0]):
                    self.weights[l] - self.learning_rate * (
                        activations[l][n].reshape((self.structure[l], 1)).dot(deltas[l][n].reshape((1, deltas[l][n].shape[0]))) +
                        self.momentum * self.weights[l])

    def deltas(self, activations, targets):
        """
        Find the deltas for each layer from 2 to l.

        Returns a (l-1, dl) matrix, where l is the number of layers, and dl is
            the dimension of layer l.

        activations: (l-1, n, dl) matrix, n is the number of data points.
        """

        deltas = []
        for l in range(self.layers-2, -1, -1):
            if l == self.layers - 2:
                delta = (activations[l+1] - targets) * activations[l+1] * (1 -
                    activations[l+1])
            else:
                delta = deltas[-1].dot(self.weights[l+1].T) * activations[l+1] * (1 -
                    activations[l+1])
            deltas.append(delta)
        deltas.reverse()
        return deltas

    def activations(self, data):
        """
        Find the activations for each layer from 0 to l-1.

        Returns a (l, n, dl) matrix, where l is the number of layers, n is the
        number of data points, and dl is the dimension of layer l.

        data: (n, d) matrix, where n is the number of data points and d is the
            input dimension.
        """

        activations = []
        for l in range(self.layers):
            if l == 0:
                # Input layer.
                activations.append(data)
            else:
                activation = sigmoid(activations[-1].dot(self.weights[l-1]) +
                    self.biases[l-1])
                activations.append(activation)

        return activations

def make_data(N, M):
    r = 10
    xs = np.linspace(0, r, N)
    ys = np.sin(xs-r//2) + np.random.normal(0, size=(N,))
    xs = np.reshape(xs, xs.shape + (1,))
    ys = np.reshape(ys, ys.shape + (1,))
    return xs, ys

if __name__ == '__main__':
    N = 1000
    structure = (1, 10, 1)
    nn = NeuralNetwork(structure, 0.2, 0)
    xs, ys = make_data(N, structure[0])
    nn.train(xs, ys, 100)
    plt.plot(xs, nn.activations(xs)[-1])
    plt.plot(xs, ys, "k+")
    plt.show()
    # plt.cla()
    # plt.draw()