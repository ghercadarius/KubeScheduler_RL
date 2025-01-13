import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Environment simulation
class KubernetesEnv:
    def __init__(self, num_nodes=5, readFile=False):
        if readFile:
            testFile = open("testData.txt", "r")
            self.num_nodes = int(testFile.readline())
            self.nodes = []
            for i in range(self.num_nodes):
                self.nodes.append({
                    'cpu': float(testFile.readline()),
                    'gpu': float(testFile.readline()),
                    'ram': float(testFile.readline()),
                    'response_time': 0
                })
            self.num_pods = int(testFile.readline())
            self.pods = []
            for i in range(self.num_pods):
                self.pods.append(KubernetesPod())
                self.pods[i].cpu = float(testFile.readline())
                self.pods[i].gpu = float(testFile.readline())
                self.pods[i].ram = float(testFile.readline())
                self.pods[i].response_time_factor = float(testFile.readline())
            testFile.close()
            self.response_timeout = 5
        else:
            self.num_nodes = num_nodes
            self.response_timeout = 5
            self.reset()

    def reset(self):
        self.nodes = [self._generate_node_resources() for _ in range(self.num_nodes)]
        self.pods = [KubernetesPod() for _ in range(10)]
        return self._get_state()

    def _generate_node_resources(self):
        return {
            'cpu': random.uniform(0, 0.5), # 1 - represents the maximum number of cores a node can have in our env
            'gpu': random.uniform(0, 0.5), # 1 - represents the maximum number of GPU compute cores a node can have in our env
            'ram': random.uniform(0, 0.5), # 1 - represents the maximum amount of RAM a node can have in our env
            'response_time': 0 # we start with 0 as we don't have any process
        }

    def _get_state(self):
        state = []
        for node in self.nodes:
            state.extend([node['cpu'], node['gpu'], node['ram'], node['response_time']])
        # get a random pod
        self.podClass = self.pods[random.randint(0, len(self.pods) - 1)]
        self.pod = self.podClass.getState()
        state.extend([self.pod['cpu'], self.pod['gpu'], self.pod['ram']])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        node = self.nodes[action]
        if (node['cpu'] + self.pod['cpu'] <= 1 and
            node['gpu'] + self.pod['gpu'] <= 1 and
            node['ram'] + self.pod['ram'] <= 1 and
            node['response_time'] + self.pod['response_time_factor'] <= self.response_timeout):
            node['cpu'] += self.pod['cpu']
            node['gpu'] += self.pod['gpu']
            node['ram'] += self.pod['ram']
            node['response_time'] += self.pod['response_time_factor']
            self.podClass.assigned_node.append(action)
            reward = (self.response_timeout - node['response_time']) * (1 - node['cpu']) * (1 - node['gpu']) * (1 - node['ram'])
        else:
            done = True
            for node in self.nodes:
                if not (node['cpu'] + self.pod['cpu'] > 1 or node['gpu'] + self.pod['gpu'] > 1 or node['ram'] + self.pod['ram'] > 1):
                    done = False
            if done:
                return self._get_state(), 0, True, {}
            reward = -5
        return self._get_state(), reward, False, {}

    def __str__(self):
        return f"Nodes: {self.nodes}\nPod: {self.pod}"

class KubernetesPod:
    def __init__(self):
        self.cpu = random.uniform(0, 0.5)
        self.gpu = random.uniform(0, 0.5)
        self.ram = random.uniform(0, 0.5)
        self.response_time_factor = random.uniform(0, 0.6)
        self.assigned_node = []

    def __str__(self):
        return f"CPU: {self.cpu:.2f}, GPU: {self.gpu:.2f}, RAM: {self.ram:.2f}"

    def getState(self):
        return {
            'cpu': self.cpu,
            'gpu': self.gpu,
            'ram': self.ram,
            'response_time_factor': self.response_time_factor,
            'assigned_node': self.assigned_node
        }

    def getAssignedNode(self):
        return self.assigned_node

class DefaultScheduler:
    def __init__(self, _env):
        self.env = _env

    def schedule(self, pod, classPod):
        self.actNodes = []
        for i, el in enumerate(self.env.nodes):
            self.actNodes.append((i, el))
        self.actNodes.sort(key=lambda x: x[1]['response_time'] + x[1]['cpu'] + x[1]['gpu'] + x[1]['ram'])
        for actNode in self.actNodes:
            node = actNode[1]
            indice = actNode[0]
            if (node['cpu'] + pod['cpu'] <= 1 and
                node['gpu'] + pod['gpu'] <= 1 and
                node['ram'] + pod['ram'] <= 1 and
                node['response_time'] + pod['response_time_factor'] <= self.env.response_timeout):
                self.env.nodes[indice]['cpu'] += pod['cpu']
                self.env.nodes[indice]['gpu'] += pod['gpu']
                self.env.nodes[indice]['ram'] += pod['ram']
                self.env.nodes[indice]['response_time'] += pod['response_time_factor']
                classPod.assigned_node.append(indice)
                return indice
        return -1

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

option = input("Enter 1 for training and saving or 2 for loading and testing:")
if option == "1":
    nameFile = input("Enter the name of the file to save the model:")
    nameFile = nameFile + ".pth"
    # Initialize Kubernetes environment
    env = KubernetesEnv(num_nodes=5)
    state_size = len(env.reset())  # State space size
    action_size = env.num_nodes  # Action space size
    agent = DQNAgent(state_size, action_size)
    episodes = 500  # Total training episodes
    batch_size = 32  # Batch size for experience replay
    # Training loop
    for e in range(episodes):
        state = env.reset()  # Reset environment at start of each episode
        for time in range(50):  # Max steps per episode
            action = agent.act(state)  # Agent selects action
            next_state, reward, done, _ = env.step(action)  # Execute action
            # print data about the env
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Move to next state
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break  # End episode if done
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Train the agent
        for pod in env.pods:
            print(pod.getAssignedNode(), sep=" ")
        print()
    torch.save(agent.model.state_dict(), nameFile)
elif option == "2":
    nameFile = input("Enter the name of the file to load the model:")
    nameFile = nameFile
    # Initialize Kubernetes environment
    env = KubernetesEnv(num_nodes=5, readFile=True)
    # GET SAMPLE DATA
    # testFile = open("testData.txt", "w")
    # testFile.write(len(env.nodes).__str__() + "\n")
    # for node in env.nodes:
    #     testFile.write(node['cpu'].__str__() + "\n")
    #     testFile.write(node['gpu'].__str__() + "\n")
    #     testFile.write(node['ram'].__str__() + "\n")
    # testFile.write(len(env.pods).__str__() + "\n")
    # for pod in env.pods:
    #     testFile.write(pod.cpu.__str__() + "\n")
    #     testFile.write(pod.gpu.__str__() + "\n")
    #     testFile.write(pod.ram.__str__() + "\n")
    #     testFile.write(pod.response_time_factor.__str__() + "\n")
    # testFile.close()
    #
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(nameFile))
    agent.model.eval()
    # Testing the trained agent
    state = env.reset()
    # copy values to test against default scheduler
    print("DQN Scheduler")
    for time in range(100):  # Test for 20 steps
        # print state
        action = agent.act(state)  # Agent selects action
        next_state, reward, done, _ = env.step(action)  # Execute action
        print(f"Step {time}: Action {action}, Reward {reward}")
        state = next_state  # Update state
        if done:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
            print("Nodes:")
            for node in env.nodes:
                print(node)
            break
    print("Default Scheduler")
    env = KubernetesEnv(num_nodes=5, readFile=True)
    default_scheduler = DefaultScheduler(env)
    for time in range(100):
        classPod = env.pods[random.randint(0, len(env.pods) - 1)]
        env.pod = classPod.getState()
        action = default_scheduler.schedule(env.pod, classPod)
        print("Binded pod to node: ", action)
        if action == -1:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
            print("Nodes:")
            for node in env.nodes:
                print(node)
            break