import random
from datetime import datetime

import numpy as np
import pickle

# Environment simulation
class KubernetesEnv:
    def __init__(self, num_nodes=5, readFile=False):
        if readFile:
            self.read_from_file = True
            testFile = open("testDataQLearning.txt", "r")
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
            self.pod_order = list(map(int, testFile.readline().split()))
            self.actual_pod_index = 0
            testFile.close()
            self.response_timeout = 5
        else:
            self.read_from_file = False
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
        if self.read_from_file:
            self.podClass = self.pods[self.pod_order[self.actual_pod_index]]
            self.actual_pod_index += 1
        else:
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

# Agent using DQN for learning
class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate for optimizer
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

option = input("Enter 1 for training and saving, 2 for loading and testing\nor 3 for generating new test scenario:")
if option == "1":
    nameFile = input("Enter the name of the file to save the model:")
    nameFile = nameFile + ".pkl"
    # Initialize Kubernetes environment
    env = KubernetesEnv(num_nodes=5)
    state_size = len(env.reset())  # State space size
    action_size = env.num_nodes  # Action space size
    agent = QAgent(state_size, action_size)
    episodes = 500  # Total training episodes
    # Training loop
    for e in range(episodes):
        state = env.reset()  # Reset environment at start of each episode
        for time in range(50):  # Max steps per episode
            action = agent.act(state)  # Agent selects action
            next_state, reward, done, _ = env.step(action)  # Execute action
            # print data about the env
            agent.learn(state, action, reward, next_state, done)  # Store experience
            state = next_state  # Move to next state
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break  # End episode if done
        for pod in env.pods:
            print(pod.getAssignedNode(), sep=" ")
        print()
    with open(nameFile, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("Saved agent")
elif option == "2":
    nameFile = input("Enter the name of the file to load the model:")
    nameFile = nameFile
    # Initialize Kubernetes environment
    env = KubernetesEnv(num_nodes=5, readFile=True)
    state_size = len(env.reset())
    action_size = env.num_nodes
    agent = QAgent(state_size, action_size)
    with open(nameFile, 'rb') as f:
        agent.q_table = pickle.load(f)
    print("Loaded agent")
    # Testing the trained agent
    state = env.reset()
    # get current date and time in the format: YYYYMMDDHHMMSS
    resultsName = datetime.now().strftime("%Y%m%d%H%M%S") + "QLres.txt"
    testResults = open(resultsName, "w")
    # copy values to test against default scheduler
    print("QLearning Scheduler")
    testResults.write("QLearning Scheduler\n")
    for time in range(100):  # Test for 20 steps
        # print state
        action = agent.act(state)  # Agent selects action
        next_state, reward, done, _ = env.step(action)  # Execute action
        print(f"Step {time}: Action {action}, Reward {reward}")
        testResults.write(f"Step {time}: Action {action}, Reward {reward}\n")
        state = next_state  # Update state
        if done:
            for pod in env.pods:
                print(pod.getAssignedNode(), sep=" ")
                testResults.write(pod.getAssignedNode().__str__() + "\n")
            testResults.write("Nodes:\n")
            print("Nodes:")
            for node in env.nodes:
                testResults.write(node.__str__() + "\n")
                print(node)
            break
    testResults.write("Default Scheduler\n")
    print("Default Scheduler")
    env = KubernetesEnv(num_nodes=5, readFile=True)
    default_scheduler = DefaultScheduler(env)
    for time in range(100):
        classPod = env.pods[random.randint(0, len(env.pods) - 1)]
        env.pod = classPod.getState()
        action = default_scheduler.schedule(env.pod, classPod)
        testResults.write("Binded pod to node: " + action.__str__() + "\n")
        print("Binded pod to node: ", action)
        if action == -1:
            for pod in env.pods:
                testResults.write(pod.getAssignedNode().__str__() + "\n")
                print(pod.getAssignedNode(), sep=" ")
            testResults.write("Nodes:\n")
            print("Nodes:")
            for node in env.nodes:
                testResults.write(node.__str__() + "\n")
                print(node)
            break
    testResults.close()
elif option == "3":
    env = KubernetesEnv(num_nodes=5)
    # GET SAMPLE DATA
    testFile = open("testDataQLearning.txt", "w")
    testFile.write(len(env.nodes).__str__() + "\n")
    for node in env.nodes:
        testFile.write(node['cpu'].__str__() + "\n")
        testFile.write(node['gpu'].__str__() + "\n")
        testFile.write(node['ram'].__str__() + "\n")
    testFile.write(len(env.pods).__str__() + "\n")
    for pod in env.pods:
        testFile.write(pod.cpu.__str__() + "\n")
        testFile.write(pod.gpu.__str__() + "\n")
        testFile.write(pod.ram.__str__() + "\n")
        testFile.write(pod.response_time_factor.__str__() + "\n")
    for ind in range(100):
        randIndex = random.randint(0, len(env.pods) - 1)
        testFile.write(randIndex.__str__() + " ")
    testFile.write("\n")
    testFile.close()