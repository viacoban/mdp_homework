from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from random import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Agent(object):

    def next_action(self, state):
        pass

    def observe_reward(self, reward, state, action, next_state, next_action):
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass


class RandomAgent(Agent):

    def __init__(self, action_space):
        self.action_space = action_space

    def next_action(self, state):
        return self.action_space.sample()


class MonteCarloAgent(Agent):

    def __init__(self, action_space, epsilon=0.1, discount=1.0):
        self.action_space = action_space
        self.epsilon = epsilon
        self.discount = discount
        self.trajectory = []
        self.n = defaultdict(lambda: 0)
        self.q = defaultdict(lambda: 0.0)

    def next_action(self, state):
        return self.action_space.sample() if random() < self.epsilon else (
            max(range(self.action_space.n), key=lambda a: self.q[(state, a)]))

    def observe_reward(self, reward, state, action, next_state, next_action):
        self.trajectory.append((state, action, reward))

    def start_episode(self):
        self.trajectory.clear()

    def end_episode(self):
        tail = self.trajectory
        while tail:
            g = sum([r * pow(self.discount, i) for i, (_, _, r) in enumerate(tail)])
            (state, action, _), *tail = tail
            key = (state, action)
            self.n[key] += 1
            self.q[key] += 1.0 / self.n[key] * (g - self.q[key])


class SarsaAgent(Agent):

    def __init__(self, action_space, lmbda, epsilon=0.1, discount=1.0):
        self.action_space = action_space
        self.discount = discount
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.trace = defaultdict(lambda: 0.0)
        self.n = defaultdict(lambda: 0)
        self.q = defaultdict(lambda: 0.0)

    def next_action(self, state):
        return self.action_space.sample() if random() < self.epsilon else (
            max(range(self.action_space.n), key=lambda a: self.q[(state, a)]))

    def observe_reward(self, reward, state, action, next_state, next_action):
        td_error = reward + self.discount * self.q[(next_state, next_action)] - self.q[(state, action)]
        self.trace[(state, action)] += 1
        self.n[(state, action)] += 1
        for (s, a), trace_value in self.trace.items():
            self.q[(s, a)] += 1.0 / self.n[(s, a)] * td_error * trace_value
            self.trace[(s, a)] *= self.discount * self.lmbda

    def start_episode(self):
        self.trace.clear()


class CoarseCoding(object):

    def __init__(self, player_buckets, dealer_buckets):
        self.player_buckets = player_buckets
        self.dealer_buckets = dealer_buckets
        self.shape = (len(player_buckets), len(dealer_buckets), 2, 2)

    def get_size(self):
        return len(self.player_buckets) * len(self.dealer_buckets) * 2 * 2

    def transform(self, state, action):
        player, dealer, has_ace = state
        result = np.zeros(self.shape, dtype=float)
        player_idx = [i for i, (l, u) in enumerate(self.player_buckets) if l <= player <= u]
        dealer_idx = [i for i, (l, u) in enumerate(self.dealer_buckets) if l <= dealer <= u]
        ace_idx = 1 if has_ace else 0
        action_idx = action
        player_slice = slice(player_idx[0], player_idx[-1] + 1)
        dealer_slice = slice(dealer_idx[0], dealer_idx[-1] + 1)
        # print(player_idx, dealer_idx, ace_idx, action_idx)
        result[player_slice, dealer_slice, ace_idx, action_idx] = 1
        return result


class FunctionalSarsaAgent(Agent):

    def __init__(self, action_space, coding, lambda_, epsilon=0.05, discount=1.0, step_size=0.01):
        self.action_space = action_space
        self.coding = coding
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.discount = discount
        self.step_size = step_size
        self.theta = np.zeros(self.coding.shape, dtype=np.float32)
        self.trace = np.zeros(self.coding.shape, dtype=np.float32)

    def next_action(self, state):
        return self.action_space.sample() if random() < self.epsilon else (
            max(range(self.action_space.n), key=lambda a: np.sum(self.theta * self.coding.transform(state, a))))

    def observe_reward(self, reward, state, action, next_state, next_action):
        w_t = self.coding.transform(state, action)
        w_t_next = self.coding.transform(next_state, next_action)
        td_error = reward + self.discount * np.sum(self.theta * w_t_next) - np.sum(self.theta * w_t)
        self.trace = self.trace * self.discount * self.lambda_ + w_t
        self.theta = self.theta + self.step_size * td_error * self.trace

    def start_episode(self):
        self.trace.fill(0)


class AgentStats(object):

    def __init__(self):
        self.value = defaultdict(lambda: 0.0)
        self.n = defaultdict(lambda: 0)

    def increment(self, state, reward):
        player, dealer, _ = state
        self.n[(player, dealer)] += 1
        self.value[(player, dealer)] += 1 / self.n[(player, dealer)] * (reward - self.value[(player, dealer)])


####################################################################################################################
def plot_rewards(stats):
    df = pd.DataFrame(stats)
    df.cumsum().plot(x='episode', y='reward', kind='line')
    plt.show()


def plot_value_function(stats):
    player_cards = [player for (player, _) in stats.value.keys()]
    dealer_cards = [dealer for (_, dealer) in stats.value.keys()]

    min_dealer_card = min(dealer_cards)
    x = np.arange(min_dealer_card, max(dealer_cards) + 1, 1, dtype=np.int32)

    min_player_card = min(player_cards)
    y = np.arange(min_player_card, max(player_cards) + 1, 1, dtype=np.int32)

    x, y = np.meshgrid(x, y)
    rewards = [stats.value[(player, dealer)] for dealer, player in zip(np.ravel(x), np.ravel(y))]
    z = np.array(rewards).reshape(x.shape)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_zlim(-1.0, 1.0)
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel('dealer card')
    ax.set_ylabel('player sum')
    plt.show()


####################################################################################################################
def run_agent(episode_count, agent_fn):
    env = gym.make('Blackjack-v0')
    agent = agent_fn(env.action_space)
    stats = AgentStats()
    for episode in range(episode_count):
        state = env.reset()
        agent.start_episode()
        is_done = False
        action = agent.next_action(state)
        while not is_done:
            next_state, reward, is_done, _ = env.step(action)
            next_action = agent.next_action(next_state)
            stats.increment(state, reward)
            agent.observe_reward(reward, state, action, next_state, next_action)
            state = next_state
            action = next_action
        agent.end_episode()
    plot_value_function(stats)


def main(n):
    coding = CoarseCoding([(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21), (22, 50)], [(1, 4), (4, 7), (7, 10)])
    # run_agent(n, MonteCarloAgent)
    # run_agent(n, lambda action_space: SarsaAgent(action_space, 0.9))
    run_agent(n, lambda action_space: FunctionalSarsaAgent(action_space, coding, 0.9))


if __name__ == '__main__':
    main(500_000)
