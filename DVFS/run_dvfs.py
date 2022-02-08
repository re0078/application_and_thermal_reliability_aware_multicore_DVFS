import numpy as np

from agent import DVFSAgent
from power_prediction import PowerPredictor


def run():
    initial_loop = int(1e6)
    secondary_loop = int(1e7)
    power_predictor = PowerPredictor()
    agent = DVFSAgent(power_predictor)
    state = agent.get_initial_state()
    power_hist = [0]
    for i in range(initial_loop):
        new_state, _, _ = agent.env_random_step(state)
        p = new_state[0]
        power_predictor.partial_fit(np.array(power_hist), p)

    Q, stat = agent.start(secondary_loop)
    with open("out", 'w') as f:
        f.write(stat)


def __main__():
    run()
