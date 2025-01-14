import random

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