"""
Neural network, using Theano.

Matthew Alger
2015
"""

class NeuralNetwork(object):

    """
    Neural network with a varying number of layers.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate