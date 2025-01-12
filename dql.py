import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Environment simulation
class KubernetesEnv:
    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        self.pods = [KubernetesPod(random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5)) for _ in range(10)]
        self.reset()

    def reset(self):
        self.nodes = [self._generate_node_resources() for _ in range(self.num_nodes)]
        return self._get_state()

    def _generate_node_resources(self):
        return {
            'cpu': random.uniform(0, 1),
            'gpu': random.uniform(0, 1),
            'ram': random.uniform(0, 1),
            'response_time': random.uniform(0, 1)
        }

    def _get_state(self):
        state = []
        for node in self.nodes:
            state.extend([node['cpu'], node['gpu'], node['ram'], node['response_time']])
        # get a random pod
        self.pod = self.pods[random.randint(0, len(self.pods) - 1)].getState()
        state.extend([self.pod['cpu'], self.pod['gpu'], self.pod['ram']])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        node = self.nodes[action]
        if (node['cpu'] + self.pod['cpu'] <= 1 and
            node['gpu'] + self.pod['gpu'] <= 1 and
            node['ram'] + self.pod['ram'] <= 1):
            node['cpu'] += self.pod['cpu']
            node['gpu'] += self.pod['gpu']
            node['ram'] += self.pod['ram']
            reward = 1 - self.pod['response_time_factor']
        else:
            reward = -1
        self.pod = self.pods[random.randint(0, len(self.pods) - 1)].getState()
        return self._get_state(), reward, False, {}

    def __str__(self):
        return f"Nodes: {self.nodes}\nPod: {self.pod}"

class KubernetesPod:
    def __init__(self, cpu, gpu, ram):
        self.random_cpu_factor = random.uniform(0, 0.2)
        self.random_gpu_factor = random.uniform(0, 0.2)
        self.random_ram_factor = random.uniform(0, 0.2)
        self.cpu = min(1, cpu + self.random_cpu_factor)
        self.gpu = min(1, gpu + self.random_gpu_factor)
        self.ram = min(1, ram + self.random_ram_factor)
        self.response_time_factor = random.uniform(0, 0.6)

    def __str__(self):
        return f"CPU: {self.cpu:.2f}, GPU: {self.gpu:.2f}, RAM: {self.ram:.2f}"

    def getState(self):
        return {
            'cpu': self.cpu,
            'gpu': self.gpu,
            'ram': self.ram,
            'response_time_factor': self.response_time_factor
        }


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
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
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate for optimizer
        self.model = DQN(state_size, action_size)  # Q-Network
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
# Initialize Kubernetes environment
env = KubernetesEnv(num_nodes=5)
state_size = len(env.reset())  # State space size
action_size = env.num_nodes  # Action space size
agent = DQNAgent(state_size, action_size)
episodes = 100  # Total training episodes
batch_size = 32  # Batch size for experience replay

# Training loop
for e in range(episodes):
    state = env.reset()  # Reset environment at start of each episode
    for time in range(100):  # Max steps per episode
        action = agent.act(state)  # Agent selects action
        next_state, reward, done, _ = env.step(action)  # Execute action
        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state  # Move to next state
        if done:
            print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
            break  # End episode if done
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # Train the agent

# Testing the trained agent
state = env.reset()
for time in range(20):  # Test for 20 steps
    # print state
    print(f"State {time}: {state}")
    action = agent.act(state)  # Agent selects action
    next_state, reward, done, _ = env.step(action)  # Execute action
    print(f"Step {time}: Action {action}, Reward {reward}")
    state = next_state  # Update state
    if done:
        break
