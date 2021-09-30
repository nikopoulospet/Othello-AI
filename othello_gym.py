import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
from npBoard import npBoard
from random_agent import random_agent

class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 player1=None,
                 player2=None):
        super(OthelloEnv, self).__init__()

        #define observation space and action space, arrays of 64 elements for the 64 squares in othello
        self.observation_space = spaces.Box(low = -1 * np.ones(64), high = np.ones(64), dtype=np.int8)
        self.action_space = spaces.Discrete(64)
        self.Board = npBoard()

        self.player1 = player1
        self.player2 = player2

        self.max_reward = 999999


    def step(self, action):
        '''
        update state: returns an observation, reward, done, and debugging info
        input: action is a discrete value 0-63 that represents the move index
        '''
        nextboard, done, _ = self.step_player(action, self.player1, self.Board.board)

        if done:
            # invalid move selected
            return nextboard, self.max_reward * -1, done, {}
        if self.game_over(nextboard):
            print("game over due to no moves remaining")
            return nextboard, self.calc_winner(nextboard), True, {}

        # eval rewards based on opponents move
        p2_reward = self.calculate_reward(nextboard)

        nextboard *= -1
        nextboard, done, _ = self.step_player(None, self.player2, nextboard)
        nextboard *= -1

        if done:
            # invalid move selected
            return nextboard, self.max_reward, done, {}
        if self.game_over(nextboard):
            print("game over due to no moves remaining")
            return nextboard, self.calc_winner(nextboard), True, {}

        # evaluate reward based on opponents move
        p1_reward = self.calculate_reward(nextboard)
        return np.array(nextboard, dtype=np.int8), p1_reward, done, {}

    def step_player(self, action, player, board_state):
        # do valid checking and calc resulting board state for each move taken
        valid_moves = npBoard.getLegalmoves(1, board_state)
        if not action:
            action = player.get_action(board_state)

        if action not in valid_moves:
            # no valid move chosen so end the game
            print("made an invalid move, move: {}, Valid: {}".format(action, valid_moves))
            done = True
            nextboard = board_state
        else:
            # update board
            nextboard = npBoard.set_piece_index(action, 1, board_state)
            self.Board.board = nextboard
            done = False
        return nextboard, done, {}

    def game_over(self, board):
        return False

    def calc_winner(self, nextboard):
        '''
        calculates winner and returns appropriate reward
        '''
        sign = np.sum(nextboard)
        return self.max_reward * (sign/abs(sign))

    def calculate_reward(self, nextBoard):
        '''
        currently just the eval heuristic from agent.py
        '''
        #if 64 - np.sum(np.abs(nextBoard)) <= DEPTH_LIMIT * 2:
        #    return np.sum(nextBoard)

        ourLegalMoves = len(npBoard.getLegalmoves(1, nextBoard))
        theirLegalMoves = len(npBoard.getLegalmoves(-1, nextBoard))
        moveWeight = ourLegalMoves - theirLegalMoves

        discWeight = np.sum(nextBoard)

        spotWeights = np.array([4, -3, 2, 2, 2, 2, -3, 4,
                                -3, -4, -1, -1, -1, -1, -4, -3,
                                2, -1, 1, 0, 0, 1, -1, 2,
                                2, -1, 0, 1, 1, 0, -1, 2,
                                2, -1, 0, 1, 1, 0, -1, 2,
                                2, -1, 1, 0, 0, 1, -1, 2,
                                -3, -4, -1, -1, -1, -1, -4, -3,
                                4, -3, 2, 2, 2, 2, -3, 4])

        spotWeight = np.sum(nextBoard*spotWeights)

        return discWeight * -0.25 + spotWeight / 40 + moveWeight / 10

    def reset(self):
        # reset state to normal
        self.Board = npBoard()
        return self.Board.board

    def render(self, mode='human'):
        # print board info
        print(npBoard.to_str(self.Board.board,[]))

    def close(self):
        # close any threads, windows ect
        return

def createAgent(policy_type='random',
                rand_seed=0,
                search_depth=1):
    '''
    Agent factory to help deploy different algos for training and analysis
    '''
    if policy_type == 'random':
        policy = random_agent(rand_seed=rand_seed)
    else:
        print("yo tf you doing broski")
    return policy


def sim(player1= 'random',
        player2= 'random',
        sim_rounds = 100,
        search_depth = 1,
        rand_seed = 0,
        reward_function = None,
        render = True):
    '''
    starts the sim for playing two Agents against eachother
    '''
    print("PLAYER 1 [RED]: {}".format(player1))
    print("PLAYER 2 [BLUE]: {}".format(player2))
    print("player 1 goes first, othello mapping: red-> white, blue->black")
    Player1 = createAgent(policy_type=player1,
                          rand_seed=rand_seed,
                          search_depth=search_depth)

    Player2 = createAgent(policy_type=player2,
                          rand_seed=rand_seed,
                          search_depth=search_depth)

    env = OthelloEnv(Player1,Player2)

    wins_p1 = draw = loss_p1 = 0
    for i in range(sim_rounds):
        print('episode {}'.format(i))
        obs = env.reset()
        if render:
            env.render()
        done = False
        while not done:
            action = Player1.get_action(obs)
            obs, reward, done, _ = env.step(action)
            print("p1 reward {}".format(reward))
            if render:
                env.render()

            if done:
                print("final reward")
                print(reward)
if __name__ == "__main__":
    sim(sim_rounds=1)
