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
