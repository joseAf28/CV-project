


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}


    def add_node(self, node_id, data=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = data
            self.edges[node_id] = []


    def add_edge(self, node1_id, node2_id, homography):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.edges[node1_id].append((node2_id, homography))
            self.edges[node2_id].append((node1_id, homography))


    def get_node(self, node_id):
        return self.nodes.get(node_id)


    def get_edge(self, node1_id, node2_id):
        for edge in self.edges.get(node1_id, []):
            if edge[0] == node2_id:
                return edge
        return None
