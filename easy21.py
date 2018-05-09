from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from enum import Enum
from random import choice, random

State = namedtuple('State', ['dealer_sum', 'player_sum', 'is_terminal'])


class Action(Enum):
    STICK = 1
    HIT = 2


class Policy:
    def reward(self, reward):
        pass

    def next_action(self, state):
        pass

    def new_episode(self):
        pass


class Easy21Environment(object):

    def __init__(self, dealer, p_red=1.0/3, p_black=2.0/3):
        self.dealer = dealer
        self.cards = range(1, 11)
        self.p_red = p_red
        self.p_black = p_black
        self.traces = []

    @staticmethod
    def _is_bust(sum):
        return sum < 1 or sum > 21

    def _reward(self, state):
        if not state.is_terminal:
            return 0
        if self._is_bust(state.player_sum):
            return -1
        if self._is_bust(state.dealer_sum):
            return 1
        if state.dealer_sum > state.player_sum:
            return -1
        if state.dealer_sum < state.player_sum:
            return 1
        else:
            return 0

    def start(self):
        self.traces.clear()
        state = State(self.draw(0, 0, 1), self.draw(0, 0, 1), False)
        self.traces.append(state)
        return state

    def draw(self, sum, p_red, p_black):
        card = choice(self.cards)
        p = random()
        if p < p_red:
            return sum - card
        elif p < p_red + p_black:
            return sum + card
        else:
            raise ValueError('p > p_red + p_black (%f > %f > %f)', p, p_red, p_black)

    def next_state(self, state, player_action):
        if player_action == Action.HIT:
            s = self.draw(state.player_sum, self.p_red, self.p_black)
            state = State(state.dealer_sum, s, self._is_bust(s))
            self.traces.append(state)
            return state

        if player_action == Action.STICK:
            while True:
                if state.is_terminal:
                    self.traces.append(state)
                    return state
                dealer_action = self.dealer.next_action(state)
                if dealer_action == Action.STICK:
                    state = State(state.dealer_sum, state.player_sum, True)
                    self.traces.append(state)
                    return state
                if dealer_action == Action.HIT:
                    s = self.draw(state.dealer_sum, self.p_red, self.p_black)
                    state = State(s, state.player_sum, self._is_bust(s))

    def step(self, state, action):
        next_state = self.next_state(state, action)
        return next_state, self._reward(next_state)


def run_episode(dealer_policy, player_policy):
    env = Easy21Environment(dealer_policy)
    dealer_policy.new_episode()
    player_policy.new_episode()
    state = env.start()
    # trace = [state]
    while True:
        action = player_policy.next_action(state)
        state, reward = env.step(state, action)
        # trace.append((action, reward))
        # trace.append(state)
        dealer_policy.reward(-reward)
        player_policy.reward(reward)
        if state.is_terminal:
            # print(trace)
            return env.traces, reward
