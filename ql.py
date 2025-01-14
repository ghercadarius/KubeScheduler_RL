import numpy as np
import random

class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.005  # Learning rate for optimizer
        self.discount_factor = 0.95  # Discount factor for future rewards
        self.q_table = {} # Q-value table

    # Choose action using epsilon-greedy strategy
    def act(self, state):
        state = tuple(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        return np.argmax(self.q_table.get(state, np.zeros(self.action_size)))  # Exploit: best action

    def learn(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        old_value = self.q_table.get(state, np.zeros(self.action_size))[action] # Old Q-value
        next_max = 0 if done else np.max(self.q_table.get(next_state, np.zeros(self.action_size))) # Next state max Q-value
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value) # New Q-value
        q_values = self.q_table.get(state, np.zeros(self.action_size))
        q_values[action] = new_value
        self.q_table[state] = q_values
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay