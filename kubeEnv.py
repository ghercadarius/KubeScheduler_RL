import kubePod as kp
import random
import numpy as np

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
            self.deploymentManifests = []
            for i in range(self.num_pods):
                self.deploymentManifests.append(kp.KubernetesPod())
                self.deploymentManifests[i].cpu = float(testFile.readline())
                self.deploymentManifests[i].gpu = float(testFile.readline())
                self.deploymentManifests[i].ram = float(testFile.readline())
                self.deploymentManifests[i].response_time_factor = float(testFile.readline())
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
        self.deploymentManifests = [kp.KubernetesPod() for _ in range(10)]
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
            self.podClass = self.deploymentManifests[self.pod_order[self.actual_pod_index]]
            self.actual_pod_index += 1
        else:
            self.podClass = self.deploymentManifests[random.randint(0, len(self.deploymentManifests) - 1)]
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
