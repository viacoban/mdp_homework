from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, defaultdict
from random import choice, random
from itertools import groupby

State = namedtuple('State', ['dealer_sum', 'player_sum'])
Step = namedtuple('Step', ['state', 'action', 'reward'])

ACTION_STICK = 'stick'
ACTION_HIT = 'hit'


####################################################################################################################
class Policy(object):
    def next_action(self, state):
        pass


class DealerPolicy(Policy):

    def next_action(self, state):
        return ACTION_HIT if state.dealer_sum < 17 else ACTION_STICK


class EGreedyPolicy(Policy):
    def __init__(self, actions, q_fn, epsilon=0.1):
        self.epsilon = epsilon
        self.actions = actions
        self.q_fn = q_fn

    def next_action(self, state):
        return choice(self.actions) if random() < self.epsilon else (
            max(self.actions, key=lambda a: self.q_fn.evaluate(state, a)))


####################################################################################################################
class QFunction(object):

    def new_episode(self):
        pass

    def evaluate(self, state, action):
        pass

    def add_step(self, state, action, reward):
        pass

    def update(self, is_terminal, next_state, next_action):
        pass


class MonteCarloQFunction(QFunction):

    def __init__(self, discount):
        self.discount = discount
        self.n = defaultdict(lambda: 0)
        self.q = defaultdict(lambda: 0.0)
        self.trajectory = []

    def new_episode(self):
        self.trajectory.clear()

    def add_step(self, state, action, reward):
        self.trajectory.append(Step(state, action, reward))

    def evaluate(self, state, action):
        return self.q[(state, action)]

    def update(self, is_terminal, next_state, next_action):
        if not is_terminal:
            return

        tail = self.trajectory
        while tail:
            g = sum([s.reward * pow(self.discount, i) for i, s in enumerate(tail)])
            step, *tail = tail
            key = (step.state, step.action)
            self.n[key] += 1
            self.q[key] += 1.0 / self.n[key] * (g - self.q[key])


class SarsaQFunction(QFunction):

    def __init__(self, discount, lmbda):
        self.discount = discount
        self.lmbda = lmbda
        self.n = defaultdict(lambda: 0)
        self.q = defaultdict(lambda: 0.0)
        self.trace = defaultdict(lambda: 0.0)
        self.last_step = None

    def new_episode(self):
        self.trace.clear()

    def evaluate(self, state, action):
        return self.q[(state, action)]

    def add_step(self, state, action, reward):
        self.last_step = Step(state, action, reward)
        self.n[(state, action)] += 1

    def update(self, is_terminal, next_state, next_action):
        key = (self.last_step.state, self.last_step.action)
        next_key = (next_state, next_action)
        next_q = self.q[next_key] if next_key in self.q else 0
        td_error = self.last_step.reward + self.discount * next_q - self.q[key]
        self.trace[key] += 1
        for k, trace in self.trace.items():
            self.q[k] += 1.0 / self.n[k] * td_error * trace
            self.trace[k] *= self.discount * self.lmbda


####################################################################################################################
class Easy21Environment(object):

    def __init__(self, dealer, p_red=1.0/3, p_black=2.0/3):
        self.dealer = dealer
        self.cards = range(1, 11)
        self.p_red = p_red
        self.p_black = p_black

    @staticmethod
    def _is_bust(sum):
        return sum < 1 or sum > 21

    def _reward(self, state, is_terminal):
        if not is_terminal:
            return 0
        if self._is_bust(state.dealer_sum):
            return 1
        if self._is_bust(state.player_sum):
            return -1
        if state.dealer_sum > state.player_sum:
            return -1
        if state.dealer_sum < state.player_sum:
            return 1
        else:
            return 0

    def start(self):
        return State(self.draw(0, 0, 1), self.draw(0, 0, 1))

    def draw(self, value, p_red, p_black):
        card = choice(self.cards)
        p = random()
        if p < p_red:
            return value - card
        elif p < p_red + p_black:
            return value + card
        else:
            raise ValueError('p > p_red + p_black (%f > %f > %f)', p, p_red, p_black)

    def next_state(self, state, player_action):
        # terminal state is player is bust
        if self._is_bust(state.player_sum):
            return state, True
        # draw next card for player
        if player_action == ACTION_HIT:
            s = self.draw(state.player_sum, self.p_red, self.p_black)
            return State(state.dealer_sum, s), self._is_bust(s)
        # stop drawing and let dealer go
        if player_action == ACTION_STICK:
            while True:
                if self._is_bust(state.dealer_sum):
                    return state, True
                dealer_action = self.dealer.next_action(state)
                if dealer_action == ACTION_STICK:
                    return State(state.dealer_sum, state.player_sum), True
                if dealer_action == ACTION_HIT:
                    s = self.draw(state.dealer_sum, self.p_red, self.p_black)
                    state = State(s, state.player_sum)

    def step(self, state, action):
        next_state, is_terminal = self.next_state(state, action)
        return next_state, self._reward(next_state, is_terminal), is_terminal


def q_fn_to_value_fn(Q):
    q = sorted(list(Q.items()))
    result = defaultdict(lambda: 0)
    for state, items in groupby(q, lambda key: key[0][0]):
        v = max([i[1] for i in items])
        result[state] = v
    return result


def run_monte_carlo(n):
    actions = [ACTION_STICK, ACTION_HIT]
    monte_carlo = MonteCarloQFunction(discount=1.0)
    player_policy = EGreedyPolicy(actions, monte_carlo, epsilon=0.1)
    dealer = DealerPolicy()

    env = Easy21Environment(dealer)
    rewards = []
    for i in range(n):
        monte_carlo.new_episode()
        is_terminal = False
        reward = 0
        state = env.start()
        while not is_terminal:
            action = player_policy.next_action(state)
            next_state, reward, is_terminal = env.step(state, action)
            monte_carlo.add_step(state, action, reward)
            state = next_state
        monte_carlo.update(True, None, None)
        rewards.append({'episode': i, 'reward': reward})
    return rewards, monte_carlo.q


def run_sarsa(n, lmbda):
    actions = [ACTION_STICK, ACTION_HIT]
    q_fn = SarsaQFunction(discount=1.0, lmbda=lmbda)
    player_policy = EGreedyPolicy(actions, q_fn, epsilon=0.1)
    dealer = DealerPolicy()

    env = Easy21Environment(dealer)
    rewards = []
    for i in range(n):
        q_fn.new_episode()
        is_terminal = False
        reward = 0
        state = env.start()
        action = player_policy.next_action(state)
        while not is_terminal:
            next_state, reward, is_terminal = env.step(state, action)
            q_fn.add_step(state, action, reward)

            next_action = action if is_terminal else player_policy.next_action(next_state)
            q_fn.update(is_terminal, next_state, next_action)
            action = next_action
            state = next_state
        rewards.append({'episode': i, 'reward': reward})
    return rewards, q_fn.q
