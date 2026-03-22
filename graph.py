import pandas as pd
import numpy as np
import os
import time
from heapdict import heapdict
import multiprocessing
import itertools

class Graph:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.I = 0 # number of instances
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
            self.out_arcs[u].append(a)
            self.in_arcs[v].append(a)

    def load(city='sh', dir_='data'): # sh, la or sz
        g = Graph()
        g.arcs = np.loadtxt(os.path.join(dir_, f'{city}_arcs.csv'), delimiter=',', dtype=int).tolist()
        g.weights = pd.read_csv(os.path.join(dir_, f'{city}_weights.csv'), index_col=0, header=0)
        g.weights.columns = g.weights.columns.astype(int)
        g.I = len(g.weights)
        g.v_labels = np.loadtxt(os.path.join(dir_, f'{city}_labels_nodes.csv'), dtype=object).tolist()
        g.n = len(g.v_labels)
        g.m = len(g.arcs)
        g.build_out_in()
        return g
    
    def __repr__(self):
        return f'<{self.__class__.__name__}({self.n}, {self.m}) at {hex(id(self))}>'

    def floyd_warshall(self, instances='all'):
        if instances == 'all':
            distance = np.zeros((self.I, self.n, self.n))
        else:
            distance = np.zeros((len(instances), self.n, self.n))
        distance.fill(np.inf)

        for a, (u, v) in enumerate(self.arcs):
            if instances == 'all':
                w = self.weights.loc[:, a]
            else:
                w = self.weights.loc[instances, a]
            distance[:, u, v] = w
        for v in range(self.n):
            distance[:, v, v] = 0

        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    via = distance[:, i, k] + distance[:, k, j]
                    distance[:, i, j] = np.where(distance[:, i, j] > via, via, distance[:, i, j])

        return distance

    def djikstra(self, s, instance=0):
        found = set()
        dists = np.zeros((self.n))
        dists.fill(np.inf)
        heap = heapdict()
        heap[s] = 0

        while len(heap):
            v, d = heap.popitem()
            dists[v] = d
            found.add(v)
            for a in self.out_arcs[v]:
                neighbour = self.arcs[a][1]
                if neighbour in found:
                    continue
                w = self.weights.loc[instance, a]
                if neighbour not in heap or heap[neighbour] > d + w:
                    heap[neighbour] = d + w
        return dists

if __name__ == "__main__":
    g = Graph.load('sh')
    start = time.time()
    fwd = g.floyd_warshall(instances=[0])
    t_fw = time.time() - start
    # now ~1.3s from multiple instances overhead, single inst implementation was ~0.14s
    print(f'one floyd warshall took {t_fw:.3f}s')

    start = time.time()
    djd = np.zeros((g.n, g.n))
    for s in range(g.n):
        djd[s, :] = g.djikstra(s)
    t_dj = time.time() - start

    # ~0.07s
    print(f'one djikstra took {t_dj:.3f}s')

    print('doing the same thing:', np.allclose(fwd, djd))

    source = 40
    k = 15
    print(f'{k} closest nodes from {source}:')
    print(sorted(list(zip(range(g.n), fwd[0][40].tolist())), key=lambda x: x[1])[:k])

    start = time.time()
    a_fwd = g.floyd_warshall()
    t_a_fw = time.time() - start
    # ~0.0096s per instance, 14x speedup, 42s total
    print(f'all floyd warshall took {t_a_fw:.3f}s, {t_a_fw:.2f} / {g.I} = {t_a_fw / g.I:.4f}')

    start = time.time()
    pool = multiprocessing.Pool(4)
    args = itertools.product([g], range(g.n), range(g.I))
    results = pool.starmap(Graph.djikstra, args)
    a_djd = np.zeros((g.I, g.n, g.n))
    for (_, s, i), res in zip(args, results):
        a_djd[i, s, :] = res

    t_a_dj = time.time() - start

    # ~0.018s per instance, just 4x speedup, 78s total
    print(f'all djikstra took {t_a_dj:.3f}s, {t_a_dj:.2f} / {g.I} = {t_a_dj / g.I:.4f}')
