import itertools
from collections import defaultdict
from time import time_ns
from typing import List

import numpy as np
import plotting

from DVFS.models.models import ThermalReliability, ApplicationReliability
from DVFS.power_prediction import PowerPredictor

_POWER_KEY, _TR_KEY, _AR_KEY = 0, 1, 2


def createEpsilonGreedyPolicy(n_actions):
    def policyFunction(Q, state, epsilon):
        action_probabilities = np.ones(n_actions,
                                       dtype=float) * epsilon / n_actions

        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policyFunction


class DVFSAgent:
    discount_factor = 0.28
    lr = 0.72
    action_space = np.array(((None, None),
                             (None, None),
                             (None, None),
                             (None, None)))  # TODO
    k = 20  # state space size
    MAX_POWER = 10  # TODO
    MIN_POWER = 1
    alpha1 = 0.33
    alpha2 = 0.33
    alpha3 = 0.33
    power_predictor: PowerPredictor
    tr: ThermalReliability
    fr: ApplicationReliability
    start_time: int
    power_hist: List[int]

    def __init__(self, power_predictor, start_time):
        self.power_predictor = power_predictor  # pre learnt power predictor
        self.tr = ThermalReliability()
        self.fr = ApplicationReliability()
        self.start_time = start_time

    def get_reward(self, s, s_prime):
        tr_reward = s_prime[_TR_KEY] / s[_TR_KEY] - 1
        ar_reward = s_prime[_AR_KEY] / s[_AR_KEY] - 1
        power_consumption_reward = s_prime[_POWER_KEY] / s[_POWER_KEY] - 1
        return self.alpha1 * tr_reward + self.alpha2 * ar_reward + self.alpha3 * power_consumption_reward

    #  assumption: current p is added to power_hist before head
    def anticipate_s_prime(self, T, w, action):
        t = time_ns() - self.start_time
        next_p = self.power_predictor.predict(np.array(self.power_hist))
        next_tr = self.tr.get_TR(t, T)
        V, f = action
        next_r = self.fr.get_FR(w, f, V)
        return self.get_state((next_p, next_r, next_tr))

    def update_Q(self, Q, reward, s, action, s_prime):
        best_next_action = np.argmax(Q[s_prime])
        td_target = reward + self.discount_factor * Q[s_prime][best_next_action]
        td_delta = td_target - Q[s][action]
        Q[s][action] += self.lr * td_delta

    def get_state(self, s):
        v, r, tr = s
        array = np.arange(self.MIN_POWER, self.MAX_POWER)
        array = np.asarray(array)
        idx = (np.abs(array - v)).argmin()
        v = array[idx]
        return v, r, tr

    @staticmethod
    def get_initial_state():
        # TODO
        return 0, 0, 0

    def env_step(self, s, action):
        # TODO
        T = 10
        w = 1
        return (0, 0, 0), T, w

    def start(self, n_episodes, epsilon=0.1):
        n_actions = self.action_space.shape[0]
        Q = defaultdict(lambda: np.zeros(n_actions))

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(n_episodes),
            episode_rewards=np.zeros(n_episodes))

        policy = createEpsilonGreedyPolicy(n_actions)
        action = None
        for ith_episode in range(n_episodes):

            state, w, T = self.get_initial_state()

            for t in itertools.count():

                cur_epsilon = np.exp(np.log(epsilon / (t + 1)))
                Q_prime = Q.copy()
                if action:
                    anticipated_next_state = self.anticipate_s_prime(T, w, action)
                    anticipated_reward = self.get_reward(state, anticipated_next_state)
                    self.update_Q(Q_prime, anticipated_reward, state, action, anticipated_next_state)

                action_probabilities = policy(Q_prime, state, cur_epsilon)

                action = np.random.choice(np.arange(
                    len(action_probabilities)),
                    p=action_probabilities)

                next_state = self.env_step(state, action)

                self.power_hist.append(state[0])
                reward = self.get_reward(state, next_state)

                stats.episode_rewards[ith_episode] += reward
                stats.episode_lengths[ith_episode] = t

                self.update_Q(Q, reward, state, action, next_state)
                state = next_state

        return Q, stats
