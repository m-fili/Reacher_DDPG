import numpy as np
import random
from collections import deque, namedtuple


class ReplayBuffer:

    def __init__(self, buffer_length, batch_size):
        self.buffer_length = buffer_length
        self.buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.experience_tuple = namedtuple("exp_tuple", field_names=["state", "action", "reward", "next_state", "done"])

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        experience = self.experience_tuple(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_experience(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        sampled_tuples = random.sample(self.buffer, k=self.batch_size)
        states = np.vstack([t.state for t in sampled_tuples if t is not None])
        actions = np.vstack([t.action for t in sampled_tuples if t is not None])
        rewards = np.vstack([t.reward for t in sampled_tuples if t is not None])
        next_states = np.vstack([t.next_state for t in sampled_tuples if t is not None])
        dones = np.vstack([t.done for t in sampled_tuples if t is not None])
        return states, actions, rewards, next_states, dones

    def ready_to_learn(self):
        """
        Check if there are enough samples in memory to learn from.
        """
        return len(self.buffer) >= self.batch_size
