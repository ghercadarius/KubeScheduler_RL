import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np


class DQL(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQL, self).__init__()
        # First hidden layer with 128 neurons
        self.fc1 = nn.Linear(state_size, 128)
        # Second hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 128)
        # Output layer with size equal to number of possible actions
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # Forward pass with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Agent using DQN for learning
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.model = DQL(state_size, action_size)  # Q-Network
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error loss function

    # Store experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose action using epsilon-greedy strategy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: random action
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()  # Exploit: best action

    # Train the model using replay memory
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to sample from
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                # Q-learning target with discounted future reward
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target  # Update Q-value for chosen action
            self.model.train()
            output = self.model(state)
            loss = self.loss_fn(output, target_f)  # Compute loss
            self.optimizer.zero_grad()
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Gradient descent
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay