class DefaultScheduler:
    def __init__(self, env):
        self.env = env

    def schedule(self, pod):
        self.env.nodes = sorted(self.env.nodes, key=lambda x: x['cpu'] + x['gpu'] + x['ram'])
        for i, node in enumerate(self.env.nodes):
            if (node['cpu'] + pod['cpu'] <= 1 and
                node['gpu'] + pod['gpu'] <= 1 and
                node['ram'] + pod['ram'] <= 1):
                node['cpu'] += pod['cpu']
                node['gpu'] += pod['gpu']
                node['ram'] += pod['ram']
                node['response_time'] += pod['response_time_factor']
                pod['assigned_node'] = i
                return i
        return -1