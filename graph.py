import pandas as pd
import numpy as np
import os

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
        g.v_labels = np.loadtxt(os.path.join(dir_, f'{city}_labels_nodes.csv'), dtype=object).tolist()
        g.n = len(g.v_labels)
        g.m = len(g.arcs)
        g.build_out_in()
        return g

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.n}, {self.m}) at {hex(id(self))}>'


