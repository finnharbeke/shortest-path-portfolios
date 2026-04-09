import pandas as pd
import functools
import numpy as np

from graph import Graph
from path import Path
import draw

class Portfolio:
    def __init__(self, graph: Graph, paths: list[str],
                 infos: list[str] | None = None):
        """ paths is a list of path strings """
        self.g = graph
        if len(paths) == 0:
            raise ValueError('empty portfolio')
        paths = list(map(functools.partial(Path.to_arc_based, graph=graph), paths))
        self.P = paths
        self.infos = infos
        self.k = len(paths)

        self._score = None
        self._opt = None

    def __repr__(self):
        if self._score is not None and self._opt is not None:
            return f'<{self.__class__.__name__}(k={self.k}, c={self._score / self._opt:.3%}) at {hex(id(self))}>'

        return f'<{self.__class__.__name__}(k={self.k}) at {hex(id(self))}>'

    def costs(self):
        costs = pd.DataFrame(columns=self.P)
        int_P = map(Path.to_integers, self.P)
        for ixs, path in zip(int_P, self.P):
            p_costs = self.g.weights[ixs].sum(axis=1)
            costs[path] = p_costs
        return costs

    def score(self):
        if self._score is not None:
            return self._score
        self._score = self.costs().min(axis=1).mean()
        return self._score

    def draw(self, **kwargs):
        title = f'k = {self.k}'
        if self._score is not None:
            title += f', cost = {self._score:.3f}'
            if self._opt is not None:
                title += f', {self._score / self._opt:.3%} of opt'
        draw.draw_paths(self.g, self.P, info=self.infos, title=title, **kwargs)

    @staticmethod
    def most_frequent(graph: Graph, s, t, k=5) -> Portfolio:
        df = pd.DataFrame(columns=['dist', 'path'])
        for i in range(graph.I):
            df.loc[i] = graph.djikstra(s, t, instance=i, path=True)

        vc = df['path'].value_counts(normalize=True)
        paths = vc.index[:k]
        dists = vc.iloc[:k]
        dists = [f'{d:.3f}' for d in dists]

        p = Portfolio(graph, paths, dists)
        p._opt = df['dist'].mean()
        return p

    @staticmethod
    def greedy(graph: Graph, s, t, k=5) -> Portfolio:
        ps = list(graph.nx_all_paths(s, t))
        all_path_port = Portfolio(graph, ps)
        all_path_costs = all_path_port.costs()
        exp_val = all_path_costs.mean()
        # overall best path
        best_avg = exp_val.index[exp_val.argmin()]
        my_costs = all_path_costs[best_avg].copy()
        portfolio = [best_avg]
        for i in range(1, k):
            # add i-th to the group
            pass
        return Portfolio(graph, portfolio)

if __name__ == "__main__":
    g = Graph.load()

    pairs = pd.read_csv('pick_pairs/sh_picks.csv')
    u, v = pairs.iloc[0]
    print(u, v)
    p = Portfolio.greedy(g, u, v)
    print(p)
