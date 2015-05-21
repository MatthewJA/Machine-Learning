"""
Loads the Iris Plants Database.

Matthew Alger
2015
"""

import numpy

def load_data(normalised=False, centred=False):
    """
    Return the Iris Plants Database as a numpy array.

    normalised: Whether the data should be constrained between 0 and 1
        (default False). bool.
    centred: Whether the data should be centered (default False). bool.
    -> Iris Plants Database. [[float]].
    """

    with open("iris.dat") as f:
        data = numpy.genfromtxt(f, delimiter=",")

        if centred:
            mean = data.sum(axis=0)/data.shape[0]
            data -= mean

        if normalised:
            smallest = data.min(axis=0)
            largest = data.max(axis=0)
            data -= smallest
            data /= largest - smallest

        return data
