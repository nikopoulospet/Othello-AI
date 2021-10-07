import numpy as np
import random
"""
random agent class, generate moves and pick a random move
used for benchmarking
"""


class random_agent():
    def __init__(self, rand_seed=0):
        self.board = npBoard()
        self.rand_seed = rand_seed

    def get_action(self, observation: np.array([])):
        '''
        gets a random action based on an observation made about the board
        '''
        self.board.board = observation
        possible_moves = npBoard.getLegalmoves(1, self.board.board)
        if len(possible_moves) == 0:
            return -1

        return random.choice(possible_moves)
