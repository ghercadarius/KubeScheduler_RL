import random
import numpy as np

class KubernetesEnv:
    def __init__(self, num_nodes=5, response_timeout=5):
        self.num_nodes = num_nodes
        self.response_timeout = response_timeout
        self.reset()

    def reset(self):
        self.nodes = [self._generate_node_resources() for _ in range(self.num_nodes)]
        self.pods = [KubernetesPod() for _ in range(10)]
        return self._get_state()

    def _generate_node_resources(self):
        return {
            'cpu': random.uniform(0, 0.5),
            'gpu': random.uniform(0, 0.5),
            'ram': random.uniform(0, 0.5),
            'response_time': 0,
        }

    def _get_state(self):
        state = []
        for node in self.nodes:
            state.extend([node['cpu'], node['gpu'], node['ram'], node['response_time']])
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
            self.podClass.assigned_node = action
            reward = (self.response_timeout - node['response_time']) * (1 - node['cpu']) * (1 - node['gpu']) * (1 - node['ram'])
        else:
            done = True
            for node in self.nodes:
                if not (node['cpu'] + self.pod['cpu'] > 1 or node['gpu'] + self.pod['gpu'] > 1 or node['ram'] + self.pod['ram'] > 1):
                    done = False
            if done:
                return self._get_state(), 0, True, {}
            reward = -10
        return self._get_state(), reward, False, {}
    def __str__(self):
        return f"Nodes: {self.nodes}\nPod: {self.pod}"

class KubernetesPod:
    def __init__(self):
        self.cpu = random.uniform(0, 0.4)
        self.gpu = random.uniform(0, 0.4)
        self.ram = random.uniform(0, 0.4)
        self.response_time_factor = random.uniform(0, 0.6)
        self.assigned_node = None

    def getState(self):
        return {
            'cpu': self.cpu,
            'gpu': self.gpu,
            'ram': self.ram,
            'response_time_factor': self.response_time_factor,
            'assigned_node': self.assigned_node,
        }

    def getAssignedNode(self):
        return self.assigned_node
    def __str__(self):
        return f"CPU: {self.cpu:.2f}, GPU: {self.gpu:.2f}, RAM: {self.ram:.2f}"