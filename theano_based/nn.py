"""
Neural network, using Theano.

Matthew Alger
2015
"""

import time

import numpy as np
import theano as th

DEBUG = True
def debug_log(log):
    t = time.strftime("%H:%M:%S", time.localtime())
    print(t, log)

class NeuralNetwork(object):

    """
    Neural network with a varying number of layers.
    """

    def __init__(self, structure, learning_rate, reg):
        """
        structure: Tuple describing the layer dimensions, including both the
            input and output layers. E.g., (784, 100, 784) for a simple
            compression autoencoder.
        learning_rate: Rate at which the network learns.
        reg: How much the weights should be regularised.
        batch_size: Number of samples to learn from at a time.
        -> NeuralNetwork
        """

        debug_log("Initialising neural network...")

        self.structure = structure
        self.learning_rate = learning_rate
        self.reg = reg

        # Symbolic N x S[0] input matrix.
        self.s_input = th.tensor.matrix("X")
        # Symbolic N x S[-1] expected output matrix.
        self.s_expected = th.tensor.matrix("Y")

        # S[l] x S[l+1] weight matrices.
        self.weights = []
        for layer, dim in enumerate(self.structure[:-1]):
            weight_matrix = np.asarray(
                np.random.uniform(-1, 1, size=(dim, self.structure[layer+1])),
                dtype=th.config.floatX)
            self.weights.append(
                th.shared(weight_matrix,
                    name="W{}".format(layer),
                    borrow=True))

        # S[l+1] bias vectors.
        # We could also incorporate these into the weights, but while that is
        # conceptually easier, it's a lot easier to implement explicit bias
        # vectors.
        self.biases = []
        for layer, dim in enumerate(self.structure[1:]):
            bias_vector = np.asarray(
                np.random.uniform(-1, 1, size=(dim,)),
                dtype=th.config.floatX)
            self.biases.append(
                th.shared(bias_vector,
                    name="b{}".format(layer-1),
                    borrow=True))

        # Gradient descent updates.
        # Updates are tuples of the form (prev, next).
        debug_log("Finding costs...")
        cost = self.cost()
        self.weight_gradients = [th.tensor.grad(cost, w) for w in self.weights]
        self.bias_gradients = [th.tensor.grad(cost, b) for b in self.biases]
        self.updates = [(w, w-self.learning_rate*dw)
                        for w, dw in zip(self.weights, self.weight_gradients)]
        self.updates += [(b, b-self.learning_rate*db)
                         for b, db in zip(self.biases, self.bias_gradients)]
        debug_log("Found costs.")

        # Theano-compiled training function.
        debug_log("Compiling functions...")
        input_matrix = th.tensor.matrix("iX")
        expected_matrix = th.tensor.matrix("iY")
        self.train_once = th.function([input_matrix, expected_matrix],
            outputs=self.cost(),
            updates=self.updates,
            givens={
                self.s_input: input_matrix,
                self.s_expected: expected_matrix})
        self.predict = th.function([input_matrix],
            outputs=self.output,
            givens={
                self.s_input: input_matrix})
        debug_log("Compiled functions.")

        debug_log("Initialised neural network.")

    @property
    def output(self):
        """
        Get the symbolic output of the network.

        -> Symbolic output
        """

        outputs = [self.s_input]
        for layer in range(len(self.structure)-1):
            w = self.weights[layer]
            b = self.biases[layer]
            y = th.tensor.tanh(th.tensor.dot(outputs[-1], w) + b)
            outputs.append(y)

        return outputs[-1]

    def cost(self):
        """
        Get the cost of the expected output compared to the actual output.

        Uses sum-of-squares error.

        -> Symbolic cost
        """

        difference = -self.s_expected + self.output
        N = self.s_input.shape[0]
        error_cost = 1/2 * difference.T.dot(difference).sum() / N

        weights_size = sum((w**2).sum() for w in self.weights)
        W = sum(w.shape[0] * w.shape[1] for w in self.weights)
        weights_cost = self.reg * weights_size / W

        return error_cost + weights_cost

def main():
    """
    Test the neural network.
    """

    import matplotlib.pyplot as plt

    structure = (1, 10, 1)
    learning_rate = 0.01
    reg = 0.001
    N = 1000
    data = np.random.uniform(0, 1, size=(N, 1))
    targets = np.sin(data*5)
    targets = np.random.normal(targets, 0.05)
    targets -= targets.min()
    targets /= targets.max() - targets.min()

    nn = NeuralNetwork(structure, learning_rate, reg)
    trials = 1000
    batch_size = 2
    for i in range(trials):
        debug_log("Training ({}/{})".format(i+1, trials))
        for j in range(0, N, batch_size):
            nn.train_once(data[j:j+batch_size, :], targets[j:j+batch_size, :])
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation, :]
        targets = targets[permutation, :]

    plt.plot(data, targets, 'k+')
    plt.plot(data, nn.predict(data), "ro")
    plt.show()

if __name__ == '__main__':
    main()