import gym
import torch
from gym import spaces
import numpy as np
from npBoard import npBoard
from Agents.random_agent import random_agent
from Agents.agent import miniMax_agent
from Agents.processAgent import miniMaxSubOrecess_agent
from qlearning.qlearningAgent import Qagent, Experience, ReplayMemory
from qlearning.strategy import EpsilonGreedyStrategy
from qlearning.deepQNetwork import DQN, FCN
from matplotlib import pyplot as plt


class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 player1=None,
                 player2=None):
        super(OthelloEnv, self).__init__()

        # define observation space and action space, arrays of 64 elements for the 64 squares in othello
        self.observation_space = spaces.Box(low=-1 * np.ones(64), high=np.ones(64), dtype=np.int8)
        self.action_space = spaces.Discrete(64)
        gb = npBoard()
        self.board = gb.board

        self.player1 = player1
        self.player2 = player2

        self.steps = 0

        self.max_reward = 999999

    def step(self, action, board):
        '''
        update state: returns an observation, reward, done, and debugging info
        input: action is a discrete value 0-63 that represents the move index
        '''
        self.steps += 1
        nextboard, done, invalid, act = self.step_player(action, self.player1, board)
        if invalid:
            return nextboard, self.max_reward * -1, True, None
        if done:
            return nextboard, self.calc_winner(nextboard), True, None

        # eval rewards based on opponents move
        p2_reward = self.calculate_reward(nextboard)

        nextboard *= -1
        temp = torch.tensor(nextboard.astype(np.float32)).unsqueeze(0)
        nextboard, done, invalid, act = self.step_player(None, self.player2, nextboard)
        temp_next = torch.tensor(nextboard.astype(np.float32)).unsqueeze(0)
        nextboard *= -1

        if invalid:
            exp = Experience(state=temp,
                             action=torch.tensor(np.array([action]).astype(np.int64)),
                             next_state=temp_next,
                             reward=torch.tensor(np.array([self.max_reward * -1]).astype(np.float32)))
            return nextboard, 0, True, exp
        if done:
            exp = Experience(state=temp,
                             action=torch.tensor(np.array([action]).astype(np.int64)),
                             next_state=temp_next,
                             reward=torch.tensor(np.array([self.calc_winner(nextboard) * -1]).astype(np.float32)))
            return nextboard, self.calc_winner(nextboard), True, exp

        exp = Experience(state=temp,
                         action=torch.tensor(np.array([action]).astype(np.int64)),
                         next_state=temp_next,
                         reward=torch.tensor(np.array([p2_reward * -1]).astype(np.float32)))
        # evaluate reward based on opponents move
        p1_reward = self.calculate_reward(nextboard)
        self.board = np.array(nextboard)
        return np.array(nextboard, dtype=np.int8), p1_reward, done, exp

    def step_player(self, action, player, board_state):
        # do valid checking and calc resulting board state for each move taken
        valid_moves = npBoard.getLegalmoves(1, board_state)
        if not action:
            action = player.get_action(board_state)

        if action == -1  and not valid_moves == []:
            done = False
            invalid = False
            nextboard = board_state
        elif valid_moves == []:
            print("no more moves avaliable, tally winner")
            done = True
            invalid = False
            nextboard = board_state
        elif action not in valid_moves:
            # no valid move chosen so end the game
            print("made an invalid move, move: {}, Valid: {}".format(action, valid_moves))
            done = False
            invalid = True
            nextboard = board_state
        else:
            # update board
            done = False
            invalid = False
            nextboard = npBoard.set_piece_index(action, 1, board_state)

        return nextboard, done, invalid, action

    def calc_winner(self, nextboard):
        '''
        calculates winner and returns appropriate reward
        '''
        sign = np.sum(nextboard)
        if sign == 0:
            return 0

        return self.max_reward * (sign // abs(sign))

    def calculate_reward(self, nextBoard):
        '''
        currently just the eval heuristic from agent.py
        '''
        theirLegalMoves = len(npBoard.getLegalmoves(-1, nextBoard))
        score = self.steps * -2 + -2 * theirLegalMoves + 1 * np.sum(nextBoard)
        return score

    def reset(self):
        # reset state to normal
        gb = npBoard()
        self.board = gb.board
        self.steps = 0
        return self.board

    def render(self, board):
        # print board info
        print(npBoard.to_str(board, []))

    def close(self):
        # close any threads, windows ect
        return


def createAgent(policy_type='random',
                rand_seed=0,
                search_depth=1,
                eps_start=1,
                eps_end=0.01,
                decay=0.0025,
                lr=0.001):
    '''
    Agent factory to help deploy different algos for training and analysis
    '''
    if policy_type == 'random':
        policy = random_agent(rand_seed=rand_seed)
    elif policy_type == 'minimax':
        policy = miniMax_agent(search_depth=search_depth, func='norm')
    elif policy_type == 'disks':
        policy = miniMax_agent(search_depth=search_depth, func='disks')
    elif policy_type == 'process':
        policy = miniMaxSubOrecess_agent(search_depth=search_depth)
    elif policy_type == 'qagent':
        policy = Qagent(strategy=EpsilonGreedyStrategy(eps_start, eps_end, decay), num_actions=64,
                        policy_network=DQN(4, 1, 1), lr=lr, load=True)
    else:
        print("yo tf you doing broski")
    return policy


def sim(player1='random',
        player2='random',
        sim_rounds=100,
        search_depth=2,
        rand_seed=0,
        reward_function=None,
        render=True):
    '''
    starts the sim for playing two Agents against eachother
    '''

    batch_size = 256
    gamma = 0.999
    target_update = 10
    memory_size = 10000
    lr = 0.001
    memory = ReplayMemory(memory_size)

    print("PLAYER 1 [RED]: {}".format(player1))
    print("PLAYER 2 [BLUE]: {}".format(player2))
    print("player 1 goes first, othello mapping: red-> white, blue->black")
    Player1 = createAgent(policy_type=player1,
                          rand_seed=rand_seed,
                          search_depth=search_depth)
    if player1 == "qagent" and player2 == "qagent":
        Player2 = Player1
    else:
        Player2 = createAgent(policy_type=player2,
                              rand_seed=rand_seed,
                              search_depth=search_depth)

    env = OthelloEnv(Player1, Player2)

    wins_p1 = draw = loss_p1 = 0
    len_game = []
    rewards = []
    temp = 0
    for i in range(sim_rounds):
        print('episode {}'.format(i))
        obs = env.reset()
        temp = 0
        reward = 0
        if render:
            env.render(obs)
        done = False
        while not done:
            temp += reward
            action = Player1.get_action(obs)
            obs_next, reward, done, exp = env.step(action, obs)

            if player2 == 'qagent' and not exp == None:
                memory.push(exp)

            if player1 == 'qagent':
                memory.push(Experience(state=torch.tensor(obs.astype(np.float32)).unsqueeze(0),
                                       action=torch.tensor(np.array([action]).astype(np.int64)),
                                       next_state=torch.tensor(obs_next.astype(np.float32)).unsqueeze(0),
                                       reward=torch.tensor(np.array([reward]).astype(np.float32))))
                if memory.can_sample(batch_size):
                    experiences = memory.sample(batch_size)
                    Player1.calculate_loss(experiences)

            obs = obs_next

            if render:
                env.render(obs)
            if done:
                env.render(obs)
                len_game.append(env.steps)
                rewards.append(temp)
                if reward > 0:
                    print("player1 won")
                    wins_p1 += 1
                elif reward == 0:
                    print("tie")
                    draw += 1
                else:
                    print("player2 won")
                    loss_p1 += 1

        if sim_rounds % target_update == 0 and player1 == 'qagent':
            Player1.update_target_net()

    if player1 == 'qagent':
        Player1.save_model()

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(len_game)
    ax2.plot(rewards)
    plt.show()

    print("overall results")
    print("p1 wins: {}".format(wins_p1))
    print("draw: {}".format(draw))
    print("p2 wins: {}".format(loss_p1))
    print("win percent of p1 over {} games: {}".format(sim_rounds, wins_p1 / sim_rounds))


if __name__ == "__main__":
    sim(player1='qagent',
        player2='qagent',
        sim_rounds=1000,
        render=False)
