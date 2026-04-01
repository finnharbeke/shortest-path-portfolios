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
        self.weights: pd.DataFrame | None = None
        # from - edge -> to
        self.arcs = [] # arcs of form (u in 0...n-1, v in 0...n-1)
        self.out_arcs = [] # list of outgoing arcs (a in 0...m-1)
        self.in_arcs = []

    def build_out_in(self):
        self.out_arcs = []
        self.in_arcs = []
        for _ in range(self.n):
            self.out_arcs.append([])
            self.in_arcs.append([])
        for a, (u, v) in enumerate(self.arcs):
            self.out_arcs[u].append(a)
            self.in_arcs[v].append(a)

    @staticmethod
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
        assert self.weights is not None
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

    def djikstra(self, s, t=None, instance=0, path=False):
        """ returns distance array from s, if no t given, other wise dist(s, t) """
        assert self.weights is not None
        heap = heapdict()
        found = set()

        if t is None:
            dists = np.zeros((self.n,))
            dists.fill(np.inf)

        heap[s] = 0
        if path:
            paths = np.zeros((self.n,), dtype=object)
            paths.fill('')
            paths[s] = str(s)

        while len(heap):
            # next node
            v, d = heap.popitem()
            if path:
                p = paths[v]
            found.add(v)
            # found target?
            if v == t:
                if path:
                    return d, p
                else:
                    return d
            elif t is None: # track all distances
                dists[v] = d

            for a in self.out_arcs[v]:
                neighbour = self.arcs[a][1]
                if neighbour in found:
                    continue
                w = self.weights.loc[instance, a]

                if neighbour not in heap or heap[neighbour] > d + w:
                    heap[neighbour] = d + w
                    if path:
                        paths[neighbour] = p + ',' + str(neighbour)

        if path:
            if t is not None:
                return np.inf, ''
            return dists, paths
        else:
            if t is not None:
                return np.inf
            return dists

    def djikstra_all_paths(self, instance=0):
        all_paths = []
        for s in range(self.n):
            _, paths = self.djikstra(s, instance=instance, path=True)
            all_paths.append(paths)
        return np.concatenate(all_paths)

if __name__ == "__main__":
    g = Graph.load('sh')

    start = time.time()
    g.djikstra(6, path=True)
    # ~0.004s
    print(f'time for one source, all paths, {time.time() - start:.3f}s')
    start = time.time()
    g.djikstra_all_paths()
    # ~0.077s
    print(f'time for all sources, all paths, {time.time() - start:.3f}s')
    
    start = time.time()
    pool = multiprocessing.Pool(4)
    results = pool.starmap(Graph.djikstra_all_paths, itertools.product([g], range(g.I)))
    paths = pd.DataFrame(np.vstack(results), columns=[f'{u}-{v}' for u in range(g.n) for v in range(g.n)])

    t_a = time.time() - start
    print(f'time for all instances, {t_a:.3f}s, {t_a:.0f}s / {g.I} = {t_a / g.I:.3f}s')
    paths.to_csv('pick_pairs/sh_paths.csv')
