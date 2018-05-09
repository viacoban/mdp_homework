from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from easy21 import run_episode, Action
from policy import default_dealer_policy
from policy import naive_player
from policy import MonteCarlo


def plot_states(stats):
    dealer_cards = [dealer_card for (dealer_card, _) in stats.keys()]
    player_sums = [player_sum for (_, player_sum) in stats.keys()]

    min_dealer_cards = min(dealer_cards)
    x = np.arange(min_dealer_cards, max(dealer_cards) + 1, 1, dtype=np.int32)

    min_player_sum = min(player_sums)
    y = np.arange(min_player_sum, max(player_sums) + 1, 1, dtype=np.int32)

    def reward_fn(dealer_card, player_sum):
        r, c = stats[(dealer_card, player_sum)]
        return r / c if c > 0 else 0

    x, y = np.meshgrid(x, y)
    rewards = [reward_fn(dealer_card, player_sum) for dealer_card, player_sum in zip(np.ravel(x), np.ravel(y))]
    z = np.array(rewards).reshape(x.shape)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, z)
    ax.set_xlabel('dealer card')
    ax.set_ylabel('player sum')
    plt.show()


def main(n):
    stats = defaultdict(lambda: (0.0, 0))
    player = MonteCarlo(discount_factor=1, actions=[Action.STICK, Action.HIT])
    for _ in range(n):
        traces, reward = run_episode(default_dealer_policy, player)
        # print(reward, traces)
        dealer_card = traces[0].dealer_sum
        for state in traces:
            if not state.is_terminal:
                s = stats[(dealer_card, state.player_sum)]
                stats[(dealer_card, state.player_sum)] = (s[0] + reward, s[1] + 1)
    # player.update_q()
    # for i in player.trajectory:
    #     print(i)
    # for i in player.rewards.items():
    #     print('rrr', i)
    # print('zzz', player.trajectory[-1])
    # for i in player.Q.items():
    #     print('qqq', i)
    # for i in player.trajectory.items():
    #     print(i)
    plot_states(stats)


if __name__ == '__main__':
    main(100_000)
