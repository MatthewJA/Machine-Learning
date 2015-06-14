import numpy

from n_armed_bandit import (
    GreedyNArmedBandit,
    EpsilonGreedyNArmedBandit,
    GibbsNArmedBandit,

    test_bandit)


def epsilon_greedy_test():
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
        rewards = numpy.fromiter(range(10), float)
        numpy.random.shuffle(rewards)

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
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.title("$\\epsilon$-greedy n-armed bandits")
    plt.show()


def gibbs_bandit_test():
    """
    Runs bandit tests to evaluate the performance of a softmaxing n-armed
    bandit against a 0.1 epsilon-greedy n-armed bandit.
    """

    import matplotlib.pyplot as plt

    n = 10
    iterations = 1000
    e = 0.1
    temperature = 5

    trials = 500

    gibbs_received_rewards = numpy.zeros(iterations)
    gibbs2_received_rewards = numpy.zeros(iterations)
    eg_received_rewards = numpy.zeros(iterations)

    for t in range(trials):
        rewards = numpy.fromiter(range(10), float)
        numpy.random.shuffle(rewards)

        gibbs_bandit = GibbsNArmedBandit(n, temperature)
        gibbs2_bandit = GibbsNArmedBandit(n, temperature/2)
        eg_bandit = EpsilonGreedyNArmedBandit(n, e)

        gibbs_received_rewards += test_bandit(gibbs_bandit, iterations, rewards)
        gibbs2_received_rewards += test_bandit(gibbs2_bandit,
                                               iterations, rewards)
        eg_received_rewards += test_bandit(eg_bandit, iterations, rewards)

    gibbs_received_rewards /= trials
    gibbs2_received_rewards /= trials
    eg_received_rewards /= trials

    plt.plot(gibbs_received_rewards/n)
    plt.plot(gibbs2_received_rewards/n)
    plt.plot(eg_received_rewards/n)
    plt.xlabel("Iterations")
    plt.ylabel("Reward (% of maximum reward)")
    plt.title("Gibbs n-armed bandit versus epsilon-greedy n-armed bandit")
    plt.legend(["Gibbs ($\\tau = {}$)".format(temperature),
                "Gibbs ($\\tau = {}$)".format(temperature/2),
                "$\\epsilon$-greedy ($\\epsilon = {}$)".format(e)])
    plt.show()

gibbs_bandit_test()
