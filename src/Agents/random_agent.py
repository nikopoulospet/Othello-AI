import numpy as np
import random
from src.npBoard import npBoard
"""
random agent class, generate moves and pick a random move
used for benchmarking
"""


class random_agent():
    def __init__(self, rand_seed=0):
        self.rand_seed = rand_seed

    def get_action(self, observation: np.array([])):
        '''
        gets a random action based on an observation made about the board
        '''
        possible_moves = npBoard.getLegalmoves(1, observation)
        if len(possible_moves) == 0:
            return -1

        return random.choice(possible_moves)
