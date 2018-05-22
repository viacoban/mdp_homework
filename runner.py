from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from easy21 import run_monte_carlo, run_sarsa, q_fn_to_value_fn


def plot_rewards(rewards):
    df = pd.DataFrame(rewards)
    df.rolling(10000).mean().plot(x='episode', y='reward', kind='line')
    plt.show()


def plot_value_function(value_function):
    dealer_cards = [dealer_card for (dealer_card, _) in value_function.keys()]
    player_sums = [player_sum for (_, player_sum) in value_function.keys()]

    min_dealer_cards = min(dealer_cards)
    x = np.arange(min_dealer_cards, max(dealer_cards) + 1, 1, dtype=np.int32)

    min_player_sum = min(player_sums)
    y = np.arange(min_player_sum, max(player_sums) + 1, 1, dtype=np.int32)

    x, y = np.meshgrid(x, y)
    rewards = [value_function[(dealer_card, player_sum)] for dealer_card, player_sum in zip(np.ravel(x), np.ravel(y))]
    z = np.array(rewards).reshape(x.shape)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z)
    ax.set_xlabel('dealer card')
    ax.set_ylabel('player sum')
    plt.show()


def compare_mse():
    _, mc_q = run_monte_carlo(1_000_000)
    plot_value_function(q_fn_to_value_fn(mc_q))

    rows = []
    for l in (x * 0.1 for x in range(0, 10)):
        _, sarsa_q = run_sarsa(100_000, l)
        mse = 0
        for k in (mc_q.keys() | sarsa_q.keys()):
            mse += (mc_q[k] - sarsa_q[k]) ** 2
        row = {'lmbda': l, 'mse': mse}
        rows.append(row)
        print(row)
        plot_value_function(q_fn_to_value_fn(sarsa_q))

    print(rows)
    pd.DataFrame(rows).plot(x='lmbda', y='mse', kind='line')
    plt.show()


if __name__ == '__main__':
    compare_mse()
