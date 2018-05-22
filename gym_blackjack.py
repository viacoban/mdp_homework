from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from random import choice, random

import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


class AgentStats(object):

    def __init__(self):
        self.value = defaultdict(lambda:0.0)
        self.n = defaultdict(lambda:0)

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
    # plot_rewards(stats, max(episode_count / 1000, 1))
    plot_value_function(stats)


def main(n):
    run_agent(n, lambda action_space: SarsaAgent(action_space, 0.9))


if __name__ == '__main__':
    main(500_000)
