"""
Epsilon-greedy n-armed bandit agent.

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

def test_bandit(bandit, plays, rewards):
    """
    Test a bandit.

    bandit: The bandit to test. NArmedBandit.
    plays: How many plays to let the bandit have. int.
    rewards: List of average rewards for each arm. [float].
    -> List of received rewards. [float].
    """

    # Function to retreive noisy rewards - Gaussians with means of the rewards.
    noisy_rewards = lambda: numpy.random.multivariate_normal(
                                rewards,
                                numpy.identity(len(rewards)))

    # Train the bandit over a number of plays.
    received_rewards = numpy.fromiter((bandit.train(noisy_rewards())
                                       for _ in range(plays)),
                                      float)

    return received_rewards

def run_bandit_tests():
    """
    Runs bandit tests to replicate the experiment in Section 2.2 of Sutton and
    Barto's Reinforcement Learning: An Introduction.
    """

    import matplotlib.pyplot as plt

    n = 10
    iterations = 1000
    k = 0
    e = 0.1

    trials = 1000

    g_received_rewards = numpy.zeros(iterations)
    eg_received_rewards = numpy.zeros(iterations)
    eg2_received_rewards = numpy.zeros(iterations)

    for t in range(trials):
        rewards = list(range(n))

        g_bandit = GreedyNArmedBandit(n, k)
        eg_bandit = EpsilonGreedyNArmedBandit(n, e)
        eg2_bandit = EpsilonGreedyNArmedBandit(n, e/10)

        g_received_rewards += test_bandit(g_bandit, iterations, rewards)
        eg_received_rewards += test_bandit(eg_bandit, iterations, rewards)
        eg2_received_rewards += test_bandit(eg2_bandit, iterations, rewards)

    g_received_rewards /= trials
    eg_received_rewards /= trials
    eg2_received_rewards /= trials

    plt.plot(g_received_rewards)
    plt.plot(eg_received_rewards)
    plt.plot(eg2_received_rewards)
    plt.show()