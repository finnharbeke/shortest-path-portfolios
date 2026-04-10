import pandas as pd
import functools
import numpy as np
import tqdm

from graph import Graph
from path import Path
import draw

class Portfolio:
    def __init__(self, graph: Graph, paths: list[str], method='',
                 infos: list[str] | None = None):
        """ paths is a list of path strings """
        self.g = graph
        if len(paths) == 0:
            raise ValueError('empty portfolio')
        paths = list(map(functools.partial(Path.to_arc_based, graph=graph), paths))
        self.P = paths
        self.infos = infos
        self.k = len(paths)
        self.method = method

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
        title = f'{self.method} (k = {self.k})'
        if self._score is not None:
            title += f'\ncost = {self._score:.3f}'
            if self._opt is not None:
                title += f', +{self._score / self._opt - 1:.2%} to opt'
        draw.draw_paths(self.g, self.P, info=self.infos, title=title, **kwargs)

    @staticmethod
    def most_frequent(graph: Graph, s, t, k=5) -> Portfolio:
        df = pd.DataFrame(columns=['dist', 'path'])
        for i in range(graph.I):
            df.loc[i] = graph.djikstra(s, t, instance=i, path=True)

        vc = df['path'].value_counts(normalize=True)
        paths = vc.index[:k]
        freqs = vc.iloc[:k]

        p = Portfolio(graph, paths, method='most frequent shortest paths')
        p._opt = df['dist'].mean()
        costs = p.costs()
        subset_costs = costs[paths[0]]
        subset_scores = [subset_costs.mean()]
        for j in range(1, k):
            subset_costs = np.minimum(subset_costs, costs[paths[j]])
            subset_scores.append(subset_costs.mean())

        subset_ratios = [sc / p._opt - 1 for sc in subset_scores]
        infos = [f'+{r:.2%} | {f=:.3f}' for r, f in zip(subset_ratios, freqs)]
        p._score = subset_scores[-1]
        p.infos = infos
        return p

    @staticmethod
    def greedy(graph: Graph, s, t, k=5) -> Portfolio:
        ps = list(graph.nx_all_paths(s, t))
        all_path_portfolio = Portfolio(graph, ps)
        all_path_costs = all_path_portfolio.costs()
        opt = all_path_costs.min(axis=1).mean()
        expected_cost = all_path_costs.mean()
        # overall best path
        shortest_mean_path = expected_cost.index[expected_cost.argmin()]
        portfolio = [shortest_mean_path]
        costs = []
        for _ in range(1, k):
            # add the next k-1
            current_costs = all_path_costs[portfolio].min(axis=1)
            costs.append(current_costs.mean())
            # max(portfolio_cost - path_cost, 0)
            difference = -all_path_costs.sub(current_costs, axis=0)
            improvement = difference.clip(lower=0).mean()
            most_improving_path = improvement.index[improvement.argmax()]
            portfolio.append(most_improving_path)
        costs.append(all_path_costs[portfolio].min(axis=1).mean())
        infos = [f'+{c / opt - 1:.2%}' for c in costs]
        pf = Portfolio(graph, portfolio, method='greedy', infos=infos)
        pf._score = costs[-1]
        pf._opt = opt
        return pf

if __name__ == "__main__":
    g = Graph.load()

    pairs = pd.read_csv('pick_pairs/sh_picks.csv')
    pbar = tqdm.tqdm(pairs.iterrows())
    for _, (u, v) in pbar:
        pbar.write(f'{u} -> {v}')
        p = Portfolio.greedy(g, u, v, k=8)
        p.draw(savefig=f'./graph_drawings/sh/{u}-{v}_greedy.png')

    pbar = tqdm.tqdm(pairs.iterrows())
    for _, (u, v) in pbar:
        pbar.write(f'{u} -> {v}')
        p = Portfolio.most_frequent(g, u, v, k=8)
        p.draw(savefig=f'./graph_drawings/sh/{u}-{v}_greedy.png')
