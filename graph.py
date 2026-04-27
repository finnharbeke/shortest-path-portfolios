from collections import deque
import pandas as pd
import numpy as np
import os
import time
from heapdict import heapdict
import multiprocessing
import networkx as nx
import functools
import tqdm

class Graph:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.I = 0 # number of instances
        # labels for the vertices 0...n-1
        self.v_labels = None
        
        # columns are edges, rows instances, 
        self.weights: pd.DataFrame
        # from - edge -> to
        self.arcs = [] # arcs of form (u in 0...n-1, v in 0...n-1)
        self.out_arcs = [] # list of outgoing arcs (a in 0...m-1)
        self.in_arcs = []
        self.name = None
        self.w_mean = None

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
    def load(city='sh', dir_='data') -> Graph: # sh, la or sz
        g = Graph()
        arcs = np.loadtxt(os.path.join(dir_, f'{city}_arcs.csv'), delimiter=',', dtype=int).tolist()
        g.arcs = [tuple(arc) for arc in arcs]
        g.weights = pd.read_csv(os.path.join(dir_, f'{city}_weights.csv'), index_col=0, header=0)
        g.weights.columns = g.weights.columns.astype(int)
        g.I = len(g.weights)
        g.v_labels = np.loadtxt(os.path.join(dir_, f'{city}_labels_nodes.csv'), dtype=object).tolist()
        g.n = len(g.v_labels)
        g.m = len(g.arcs)
        g.build_out_in()
        g.name = city
        return g
    
    def __repr__(self):
        return f'<{self.__class__.__name__}({self.n}, {self.m}) at {hex(id(self))}>'

    def mean_weights(self):
        if self.w_mean is None:
            self.w_mean = self.weights.mean()
        return self.w_mean

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

    def djikstra(self, s, t=None, instance=0, path=False, penalties=None):
        """ returns distance array from s, if no t given, other wise dist(s, t)
            if path is True it returns tuple with path(s)
            if instance = None is passed it takes the average weight
        """
        heap = heapdict()
        found = set()

        if t is None:
            dists = np.zeros((self.n,))
            dists.fill(np.inf)

        heap[s] = 0
        if path:
            paths = np.zeros((self.n,), dtype=object)
            paths.fill('a') # a0-1-2 is arc-based path notation and v0-1-2 is vertex-based

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
                if instance is None:
                    w = self.mean_weights()[a]
                else:
                    w = self.weights.loc[instance, a]
                if penalties is not None:
                    w *= penalties[a]

                if neighbour not in heap or heap[neighbour] > d + w:
                    heap[neighbour] = d + w
                    if path:
                        paths[neighbour] = f'{p}{"-" if len(p) > 1 else ""}{a}'

        if path:
            if t is not None:
                return np.inf, ''
            return dists, paths
        else:
            if t is not None:
                return np.inf
            return dists

    def djikstra_all_pairs(self, instance=0):
        all_paths = []
        all_dists = []
        for s in range(self.n):
            dists, paths = self.djikstra(s, instance=instance, path=True)
            all_paths.append(paths)
            all_dists.append(dists)
        return np.concatenate(all_dists), np.concatenate(all_paths)

    def nx_all_paths(self, source, target):
        nxg = nx.DiGraph()
        nxg.add_edges_from((*a, dict(ix=i)) for i, a in enumerate(self.arcs))
        paths = nx.all_simple_edge_paths(nxg, source, target)
        def to_format(path):
            return 'a' + '-'.join(str(nxg.get_edge_data(u, v)['ix']) for u, v in path)
        paths = map(to_format, paths)
        return paths

    def my_all_paths(self, source, target):
        stack = deque()
        stack.append(deque([source]))
        while len(stack):
            path = stack.pop()
            u = path[-1]
            if u == target:
                yield 'v' + '-'.join(str(v) for v in path)
            for a in self.out_arcs[u]:
                v = self.arcs[a][1]
                if not v in path:
                    to_v = path.copy()
                    to_v.append(v)
                    stack.append(to_v)

if __name__ == "__main__":
    g = Graph.load('la')

    start = time.time()
    g.djikstra(6, path=True)
    # ~0.004s
    print(f'time for one source, all paths, {time.time() - start:.3f}s')
    start = time.time()
    g.djikstra_all_pairs()
    # ~0.077s
    t = time.time() - start
    print(f'time for all sources, all paths, {t:.3f}s')
    print(f'estimated time for all instances: {t*g.I:.0f}s')

    start = time.time()
    pool = multiprocessing.Pool(4)

    results = tqdm.tqdm(pool.imap(functools.partial(Graph.djikstra_all_pairs, g), range(g.I)), total=g.I)
    dists_out, paths_out = zip(*results)
    paths = pd.DataFrame(np.vstack(paths_out), columns=[f'{u}-{v}' for u in range(g.n) for v in range(g.n)])
    dists = np.vstack(dists_out)

    t_a = time.time() - start
    print(f'time for all instances, {t_a:.3f}s, {t_a:.0f}s / {g.I} = {t_a / g.I:.3f}s')
    paths.to_csv('pick_pairs/la_paths.csv')
    np.save('pick_pairs/la_dists.npy', dists)
