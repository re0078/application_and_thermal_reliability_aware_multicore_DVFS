import itertools
from collections import defaultdict

import numpy as np
import plotting
from time import time_ns
from typing import List

from DVFS.models.models import ThermalReliability, ApplicationReliability
from DVFS.power_prediction import PowerPredictor

_POWER_KEY, _TR_KEY, _AR_KEY = 0, 1, 2


def createEpsilonGreedyPolicy(Q, n_actions):
    def policyFunction(state, epsilon):
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

    def get_s_prime(self, s, T, w, ):
        t = time_ns() - self.start_time
        p, r, tr = s
        next_p = self.power_predictor.predict(np.array(self.power_hist))
        next_tr = self.tr.get_TR(t, T)
        next_r = self.fr.get_FR()


    def get_state(self, s):
        v, r, tr = s
        array = np.arange(self.MIN_POWER, self.MAX_POWER)
        array = np.asarray(array)
        idx = (np.abs(array - v)).argmin()
        v = array[idx]
        return v, r, tr

    def start(self, n_episodes, epsilon=0.1):
        n_actions = self.action_space.shape[0]
        Q = defaultdict(lambda: np.zeros(n_actions))

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(n_episodes),
            episode_rewards=np.zeros(n_episodes))

        policy = createEpsilonGreedyPolicy(Q, n_actions)

        for ith_episode in range(n_episodes):

            # state = env.reset()

            for t in itertools.count():
                pass
                # action_probabilities = policy(state)

                # action = np.random.choice(np.arange(
                #     len(action_probabilities)),
                #     p=action_probabilities)

                # next_state, reward, done, _ = env.step(action)
                #
                # stats.episode_rewards[ith_episode] += reward
                # stats.episode_lengths[ith_episode] = t
                #
                # best_next_action = np.argmax(Q[next_state])
                # td_target = reward + discount_factor * Q[next_state][best_next_action]
                # td_delta = td_target - Q[state][action]
                # Q[state][action] += alpha * td_delta

                # if done:
                #     break

                # state = next_state

        return Q, stats
