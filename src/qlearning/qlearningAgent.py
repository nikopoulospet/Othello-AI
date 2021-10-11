from random import random, randrange
import torch
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
from random import sample, choice
from src.npBoard import npBoard
import numpy as np

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, E):
        if len(self.memory) < self.capacity:
            self.memory.append(E)
        else:
            self.memory[self.push_count % self.capacity] = E
        self.push_count += 1

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size


class Qagent():
    def __init__(self, strategy, num_actions, policy_network, lr, load=False):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.gamma = 0.999

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load:
            policy_network.load_state_dict(torch.load('old_model'))
        print(policy_network)
        self.ALPHA_policy_network = policy_network.float().to(self.device)
        self.BETA_policy_network = policy_network.float().to(self.device)
        self.BETA_policy_network.load_state_dict(self.ALPHA_policy_network.state_dict())
        self.BETA_policy_network.eval()
        self.optimizer = optim.Adam(params=self.ALPHA_policy_network.parameters(), lr=lr)

    def save_model(self):
        print("saving old_model to old_model file")
        torch.save(self.BETA_policy_network.state_dict(), 'old_model')

    def update_target_net(self):
        self.BETA_policy_network.load_state_dict(self.ALPHA_policy_network.state_dict())

    @staticmethod
    def extract_tensors(experiences):
        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        return t1, t2, t3, t4

    @staticmethod
    def extract_features(states):
        '''
        extract feature boards from state
        n different feature boards -> (n,64) = new state vector output
        '''
        features = np.array([])
        for i in range(states.shape[0]):
            state = states[i, :]
            # layer 1 -> p1 pieces
            layer1 = np.zeros(64)
            layer1[np.where(state == 1)] = 1
            # layer 2 -> p2 pieces
            layer2 = np.zeros(64)
            layer2[np.where(state == -1)] = 1
            # layer 3 -> p1 valid moves
            layer3 = np.zeros(64)
            layer3[npBoard.getLegalmoves(1, state)] = 1
            # layer 4 -> p2 valid moves
            layer4 = np.zeros(64)
            layer4[npBoard.getLegalmoves(-1, state)] = 1
            # layer 5 -> all empty spots
            layer5 = np.zeros(64)
            layer5[np.where(state == 0)] = 1
            # layer 6 -> all occupied spots
            layer6 = abs(state)
            # layer 7 -> bias layer
            layer7 = np.ones(64)
            temp = np.vstack((layer1, layer2, layer3, layer4, layer5, layer6, layer7)).reshape(1, -1, 8, 8)
            if not features.any():
                features = temp
            else:
                features = np.vstack((features, temp))
        return features

    def calculate_loss(self, experiences):
        states, actions, rewards, next_states = Qagent.extract_tensors(experiences)

        current_q_values = QValues.get_current(self.ALPHA_policy_network, states, actions)
        next_q_values = QValues.get_next(self.BETA_policy_network, next_states)
        target_q_values = (next_q_values * self.gamma) + rewards

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random():
            return choice(np.append(npBoard.getLegalmoves(1, state), randrange(0, 64, 1)))  # random exploration of state space
        else:
            with torch.no_grad():
                state = Qagent.extract_features(np.expand_dims(state, axis=0))
                state = torch.tensor(state.astype(np.float32))
                state = state.to(self.device)
                return np.argmax(self.ALPHA_policy_network(state).cpu()).item()  # exploitation step


class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        states = Qagent.extract_features(states)
        states = torch.tensor(states.astype(np.float32))
        states = states.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        return policy_net(states).cpu().reshape(256, -1).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        states = Qagent.extract_features(next_states)
        states = torch.tensor(states.astype(np.float32))
        states = states.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        return target_net(states).cpu().reshape(256,-1).max(dim=1)[0].detach()
