"""
n-armed bandit agents.

Matthew Alger
2015
"""

import numpy


class NArmedBandit(object):

    """
    n-armed bandit agent.
    """

    def __init__(self, n):
        """
        n: Number of bandit arms. int.
        -> NArmedBandit.
        """

        # Number of bandit arms.
        self.n = n

        # Initialise expectations of each arm with 0.
        self.expectations = numpy.zeros(n)

        # How many plays we have had of each arm.
        self.plays = numpy.zeros(n)

        # How many plays we have had in total.
        self.total_plays = 0

    def choose_arm(self):
        """
        Return which number arm to pull.

        -> Arm pulled. int in [0, n).
        """

        raise NotImplementedError("No choose_arm function specified.")

    def train(self, rewards):
        """
        Train this bandit.

        rewards: Array of rewards. The nth element is the reward for pulling the
                 nth arm. [float].
        -> Reward. float.
        """

        # Pull an arm for a reward.
        arm = self.choose_arm()
        reward = rewards[arm]

        # Update expectations based on reward received.
        # This is just an online mean formula.
        self.expectations[arm] = self.expectations[arm]*self.plays[arm] + reward
        self.plays[arm] += 1
        self.expectations[arm] /= self.plays[arm]

        self.total_plays += 1

        return reward


class GreedyNArmedBandit(NArmedBandit):

    """
    n-armed bandit adopting a greedy strategy after some number of random pulls.
    """

    def __init__(self, n, k):
        """
        n: Number of bandit arms. int.
        k: Number of trials to pull at random before adopting a greedy strategy.
        -> GreedyNArmedBandit
        """

        # Number of trials before adopting a greedy strategy.
        self.k = k

        super().__init__(n)

    def choose_arm(self):
        """
        Return which number arm to pull, based on an initially random but later
        greedy strategy.

        -> Arm pulled. int in [0, n).
        """

        if self.total_plays < self.k:
            return numpy.random.randint(self.n)
        return self.expectations.argmax()


class EpsilonGreedyNArmedBandit(NArmedBandit):

    """
    n-armed bandit adopting an epsilon-greedy strategy.
    """

    def __init__(self, n, e):
        """
        n: Number of bandit arms. int.
        e: Chance of taking an action at random instead of adopting the greedy
           strategy.
        -> EpsilonGreedyNArmedBandit
        """

        # Chance of taking an action at random.
        self.e = e

        super().__init__(n)

    def choose_arm(self):
        """
        Return which number arm to pull. Usually picks the arm it expects will
        maximise reward, but has an epsilon chance of taking a random action
        instead.

        -> Arm pulled. int in [0, n).
        """

        if numpy.random.random() < self.e:
            return numpy.random.randint(self.n)
        return self.expectations.argmax()


class GibbsNArmedBandit(NArmedBandit):

    """
    n-armed bandit using a Gibbs distribution to decide what action to take.
    """

    def __init__(self, n, t):
        """
        n: Number of bandit arms. int.
        t: Temperature of the Gibbs distribution.
        -> GibbsNArmedBandit
        """

        # Temperature parameter of the Gibbs distribution.
        self.t = t

        super().__init__(n)

    def choose_arm(self):
        """
        Return which number arm to pull. Arms which the bandit thinks will give
        better rewards will be more likely to be pulled, according to the Gibbs
        distribution.

        -> Arm pulled. int in [0, n).
        """

        return numpy.random.multinomial(1,
                                        self.softmax(self.expectations)
                                        ).argmax()

    def softmax(self, energies):
        """
        Return an array of probabilities corresponding to energies, with higher
        energies corresponding to higher probabilities according to the Gibbs
        distribution.

        energies: Array of numbers corresponding to output probabilities.
                  [float].
        -> Probabilities. [float].
        """

        # Unnormalised probabilities.
        probabilities = numpy.exp(energies/self.t)

        # Normalisation constant (partition function).
        Z = probabilities.sum()

        return probabilities/Z


def test_bandit(bandit, plays, rewards):
    """
    Test a bandit.

    bandit: The bandit to test. NArmedBandit.
    plays: How many plays to let the bandit have. int.
    rewards: List of average rewards for each arm. [float].
    -> List of received rewards. [float].
    """

    # Function to retreive noisy rewards - Gaussians with means of the rewards.
    def noisy_rewards():
        return numpy.random.multivariate_normal(rewards,
                                                numpy.identity(len(rewards)))

    # Train the bandit over a number of plays.
    received_rewards = numpy.fromiter((bandit.train(noisy_rewards())
                                       for _ in range(plays)),
                                      float)

    return received_rewards
