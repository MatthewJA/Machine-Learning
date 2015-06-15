"""
Recurrent neural network based on
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""

import numpy as np

class RecurrentNeuralNetwork(object):

    """
    Recurrent neural network which takes some input vector and returns some
    output vector.
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Dimension parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Matrices
        self.W_past = np.random.random((self.hidden_size, self.hidden_size))
        self.W_in = np.random.random((self.hidden_size, self.input_size))
        self.W_out = np.random.random((self.output_size, self.hidden_size))

        # Layers
        self.h = np.zeros((self.hidden_size,))

    def step(self, x):
        """
        x: Input vector.
        """

        self.h = np.tanh(
            self.W_past.dot(self.h) +
            self.W_in.dot(x))

        return self.W_out.dot(self.h)

    def run(self, xs):
        """
        xs: Input vectors.
        -> inputs, targets, outputs
        """

        self.h = np.zeros((self.hidden_size,))

        inputs = []
        targets = []
        outputs = []

        for i, x in enumerate(xs):
            # Get sequential target.
            if i < len(xs) - 1:
                t = xs[i+1]
            else:
                t = np.zeros((len(x),))

            # Get output.
            y = self.step(x)
            # Softmax it.
            y = np.exp(y)/np.exp(y).sum()

            # Store the data.
            inputs.append(x)
            targets.append(t)
            outputs.append(y)

        return np.asarray(inputs), np.asarray(targets), np.asarray(outputs)

    def train(self, xs, iterations, learning_rate, momentum):
        """
        xs: Input vectors.
        iterations: Number of times to run the training algorithm.
        """

        # Forward propagation.
        inputs, targets, outputs = self.run(xs)

        # Back propagation for the past matrices.
        errors = []
        # Output layer...
        errors.append((outputs[-1] - targets[-1]) *
            outputs[-1] * (1 - outputs[-1]))
        # Other layers...
        for layer in range(xs.shape[0]-2, -1, -1):
            errors.append(self.W_past.T.dot(errors[-1]) *
                outputs[layer] * (1 - outputs[layer]))
        errors.reverse()
        # Partials...
        partials = []
        for layer in range(0, len(outputs)-1):
            partials.append(errors[layer+1].dot(outputs[layer].T))
        # Gradient descent...
        self.W_past -= learning_rate * (
            sum(partials)/len(partials) + momentum * self.W_past)

def one_of_k_encode(string):
    """
    Encodes string as 1-of-k.

    string: String to encode.
    -> [[1]], {index: character}
    """

    chars = {c: i for i, c in enumerate(sorted(set(string)))}
    inv_chars = {i: c for c, i in chars.items()}

    k = len(chars)
    encoded = np.zeros((len(string), k))
    for i, c in enumerate(string):
        j = chars[c]
        encoded[i, j] = 1
    return encoded, inv_chars

def one_of_k_decode(string, encoding):
    """
    Decodes string from 1-of-k.

    string: [[1]] to decode.
    encoding: {index: character}
    -> str
    """

    lst = []
    for x in string:
        lst.append(encoding[x.argmax()])
    return "".join(lst)

if __name__ == '__main__':
    input_string = "hello"
    input_data, encoding = one_of_k_encode(input_string)
    rnn = RecurrentNeuralNetwork(len(encoding), 3, len(encoding))
    rnn.train(input_data, 2, 0.1, 0.1)