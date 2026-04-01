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

    def score(self):
        if self._score is not None:
            return self._score
        costs = pd.DataFrame(columns=self.P)
        int_P = list(map(Path.to_integers, self.P))
        for i, row in self.g.weights.iterrows():
            i_costs = [
                row.loc[p].sum() for p in int_P
            ]
            costs.loc[i] = i_costs

        self._score = costs.min(axis=1).mean()
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
        dists = [f'{d:.3}' for d in dists]

        p = Portfolio(graph, paths, dists)
        p._opt = df['dist'].mean()
        return p

if __name__ == "__main__":
    g = Graph.load()

    pairs = pd.read_csv('pick_pairs/sh_picks.csv')
    for i, pair in pairs.iterrows():
        s, t = pair
        for k in range(1, 9):
            p = Portfolio.most_frequent(g, s, t, k=k)
            # calculate
            p.score()
            p.draw(savefig=f'graph_drawings/sh_{s}-{t}_mf{k}.png')

