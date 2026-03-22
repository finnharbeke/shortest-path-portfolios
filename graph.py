import pandas as pd
import numpy as np
import os
import time

class Graph:
    def __init__(self):
        self.n = 0
        self.m = 0
        # labels for the vertices 0...n-1
        self.v_labels = None
        
        # columns are edges, rows instances, 
        self.weights: pd.DataFrame = None
        # from - edge -> to
        self.arcs = [] # arcs of form (u in 0...n-1, v in 0...n-1)
        self.out_arcs = [] # list of outgoing arcs (a in 0...m-1)
        self.in_arcs = []

    def build_out_in(self):
        self.out_arcs = []
        self.in_arcs = []
        for u in range(self.n):
            self.out_arcs.append([])
            self.in_arcs.append([])
        for a, (u, v) in enumerate(self.arcs):
            self.out_arcs[u].append(v)
            self.in_arcs[v].append(u)

    def load(city='sh', dir_='data'): # sh, la or sz
        g = Graph()
        g.arcs = np.loadtxt(os.path.join(dir_, f'{city}_arcs.csv'), delimiter=',', dtype=int).tolist()
        g.weights = pd.read_csv(os.path.join(dir_, f'{city}_weights.csv'), index_col=0, header=0)
        g.weights.columns = g.weights.columns.astype(int)
        g.v_labels = np.loadtxt(os.path.join(dir_, f'{city}_labels_nodes.csv'), dtype=object).tolist()
        g.n = len(g.v_labels)
        g.m = len(g.arcs)
        g.build_out_in()
        return g
    
    def __repr__(self):
        return f'<{self.__class__.__name__}({self.n}, {self.m}) at {hex(id(self))}>'
    
    def floyd_warshall(self, instance=0):
        # see https://en.wikipedia.org/wiki/Floyd-Warshall_algorithm#Pseudocode
        # single instance for now
        distance = np.ones((self.n, self.n)) * np.inf
        print(self.weights.head())
        print(self.weights.index)
        print(self.weights.columns)
        for a, (u, v) in enumerate(self.arcs):
            w = self.weights.loc[instance, a]
            distance[u, v] = w
        for v in range(self.n):
            distance[v, v] = 0

        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    via = distance[i, k] + distance[k, j]
                    if distance[i, j] > via:
                        distance[i, j] = via

        return distance


if __name__ == "__main__":
    g = Graph.load('sh')
    start = time.time()
    fwd = g.floyd_warshall()
    print(f'one floyd warshall took {time.time() - start:.3f}s, {g.n}^3 = {g.n**3}')
    source = 40
    k = 15
    print(f'{k} closest nodes from {source}:')
    print(sorted(list(zip(range(g.n), fwd[40].tolist())), key=lambda x: x[1])[:k])
