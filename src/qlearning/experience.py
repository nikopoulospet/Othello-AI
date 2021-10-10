from collections import namedtuple
from random import sample

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
        return len(self.memory >= batch_size)
