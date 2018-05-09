from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from random import random, choice
from math import pow

from easy21 import Policy, Action


class NaivePolicy(Policy):

    def __init__(self, sum_fn):
        self.sum_fn = sum_fn

    def next_action(self, state):
        s = self.sum_fn(state)
        return Action.HIT if s < 17 else Action.STICK


class MonteCarlo(Policy):

    def __init__(self, discount_factor, actions):
        self.discount_factor = discount_factor
        self.actions = actions
        self.counts = defaultdict(lambda: 0)
        self.Q = defaultdict(lambda: 0.0)
        self.trajectory = []
        self.episodes = 0
        self.rewards = defaultdict(lambda: 0.0)

    def update_q(self):
        tail = self.trajectory
        while True:
            if not tail:
                break
            G = sum([self.rewards[p] * pow(self.discount_factor, i) for i, p in enumerate(tail)])
            key, *tail = tail
            self.counts[key] = self.counts[key] + 1
            self.Q[key] = self.Q[key] + 1.0 / self.counts[key] * (G - self.Q[key])

    def new_episode(self):
        self.update_q()
        self.episodes += 1.0
        self.trajectory.clear()
        self.rewards.clear()

    def reward(self, reward):
        self.rewards[self.trajectory[-1]] = reward

    def next_action(self, state):
        n = sum([self.counts[(state, a)] for a in self.actions])
        if random() > 1.0 / (100 + n):
            # if random() > 1.0 / self.episodes:
            action_q = None
            action = None
            for a in self.actions:
                q = self.Q[(state, a)]
                if not action_q or q > action_q:
                    action_q = q
                    action = a
        else:
            action = choice(self.actions)

        self.trajectory.append((state, action))
        return action


default_dealer_policy = NaivePolicy(lambda s: s.dealer_sum)
naive_player = NaivePolicy(lambda s: s.player_sum)

