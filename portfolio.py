import random
import hashlib
import inspect
import os
import pandas as pd
import functools
import numpy as np
import tqdm

from graph import Graph
from path import Path
import draw

class Portfolio:

    CACHE = './cache/'

    def __init__(self, graph: Graph, s: int, t: int, paths: list[str], method='',
                 infos: list[str] | None = None):
        """ paths is a list of path strings """
        self.g = graph
        self.s, self.t = s, t
        if len(paths) == 0:
            raise ValueError('empty portfolio')
        paths = list(map(functools.partial(Path.to_arc_based, graph=graph), paths))
        self.P = paths
        self.infos = infos
        self.k = len(paths)
        self.method = method
        self._impl_hash = ''

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

    def _to_cache(self, cache=CACHE):
        fp = Portfolio._cache_file(self.g, cache)
        fields = [
            self.s, self.t,
            self.method,
            self._impl_hash,
            self.k,
            self._score if self._score is not None else np.nan,
            self._score / self._opt if (self._opt is not None and self._score is not None) else np.nan,
            '[' + ';'.join(self.P) + ']',
            '[' + ';;'.join(self.infos) + ']' if self.infos is not None else '[]'
        ]
        if os.path.exists(fp):
            df = pd.read_csv(fp, index_col=0)
        else:
            df = pd.DataFrame(columns=['s', 't', 'method', 'hash', 'k', 'cost', 'factor', 'portfolio', 'infos'])
        if len(df) == 0:
            df.loc[0] = fields
        else:
            df.loc[df.index.max()+1] = fields
        df.drop_duplicates(inplace=True)
        df.to_csv(fp)

    @staticmethod
    def _hash(func):
        return hashlib.sha256(
            inspect.getsource(func).encode()
        ).hexdigest()

    @staticmethod
    def _cache_file(graph, cache_dir):
        if graph.name is None:
            graph.name = hex(random.randint(0x0fff, 0xf000))[2:]
        return os.path.join(cache_dir, f'{graph.name}_portfolios.csv')

    @staticmethod
    def _check_cache(graph: Graph, cache_dir, method, hash, s, t, k, is_chain=False):
        fp = Portfolio._cache_file(graph, cache_dir)
        if not os.path.exists(fp):
            return None
        pfs = pd.read_csv(fp, index_col=0)

        # check records
        same_method = (pfs['s'] == s) & (pfs['t'] == t) & (pfs['method'] == method)
        if is_chain: # can construct portfolio for k, given portfolio for k' > k
            same_method &= (pfs['k'] >= k)
        else:
            same_method &= (pfs['k'] == k)
        if same_method.sum() == 0:
            return None
        same_hash = same_method & (pfs['hash'] == hash)
        if same_hash.sum() == 0: # record exists but with legacy code, drop
            pfs.drop(index=pfs.index[same_method], inplace=True)
            pfs.to_csv(fp)
            return None

        row = pfs[same_hash].iloc[0]
        paths = row['portfolio'][1:-1].split(';')
        infos = row['infos'][1:-1].split(';;')
        chaining = False
        if k != row['k']:
            paths = paths[:k]
            infos = infos[:k]
            chaining = True
        p = Portfolio(graph, s, t, paths, method=method, infos=infos)
        p.k = k
        p._impl_hash = row['hash']
        if not pd.isna(row['factor']) and not pd.isna(row['cost']):
            p._opt = row['cost'] / row['factor']
        if chaining:
            p._score = None
            p.score()
            p._to_cache(cache=cache_dir)
        elif not pd.isna(row['cost']):
            p._score = row['cost']
        return p

    @staticmethod
    def most_frequent(graph: Graph, s, t, k=5, cache=CACHE) -> Portfolio:
        METHOD = 'most frequent shortest paths'
        HASH = Portfolio._hash(Portfolio.most_frequent)
        cached = Portfolio._check_cache(graph, cache, METHOD, HASH, s, t, k, is_chain=True)
        if cached is not None:
            return cached
        df = pd.DataFrame(columns=['dist', 'path'])
        for i in range(graph.I):
            df.loc[i] = graph.djikstra(s, t, instance=i, path=True)

        vc = df['path'].value_counts(normalize=True)
        paths = vc.index[:k]
        freqs = vc.iloc[:k]

        pf = Portfolio(graph, s, t, paths, method=METHOD)
        pf.k = k # in case the portfolio is shorter than k, keep big k as the intention
        pf._impl_hash = HASH
        pf._opt = df['dist'].mean()
        costs = pf.costs()
        subset_costs = costs[paths[0]]
        subset_scores = [subset_costs.mean()]
        for j in range(1, k):
            subset_costs = np.minimum(subset_costs, costs[paths[min(j, len(paths)-1)]])
            subset_scores.append(subset_costs.mean())

        subset_ratios = [sc / pf._opt - 1 for sc in subset_scores]
        infos = [f'+{r:.2%} | {f=:.3f}' for r, f in zip(subset_ratios, freqs)]
        pf._score = subset_scores[-1]
        pf.infos = infos
        pf._to_cache(cache=cache)
        return pf

    @staticmethod
    def greedy(graph: Graph, s, t, k=5, cache=CACHE):
        METHOD = 'greedy'
        HASH = Portfolio._hash(Portfolio.greedy)
        cached = Portfolio._check_cache(graph, cache, METHOD, HASH, s, t, k, is_chain=True)
        if cached is not None:
            return cached
        ps = []
        for p in tqdm.tqdm(graph.my_all_paths(s, t)):
            ps.append(p)
        all_path_portfolio = Portfolio(graph, s, t, ps)
        all_path_costs = all_path_portfolio.costs()
        opt = all_path_costs.min(axis=1).mean()
        expected_cost = all_path_costs.mean()
        # overall best path
        shortest_mean_path = expected_cost.index[expected_cost.argmin()]
        portfolio = [shortest_mean_path]
        costs = []
        ended_early = False
        for _ in range(1, k):
            # add the next k-1
            current_costs = all_path_costs[portfolio].min(axis=1)
            costs.append(current_costs.mean())
            # max(portfolio_cost - path_cost, 0)
            difference = -all_path_costs.sub(current_costs, axis=0)
            improvement = difference.clip(lower=0).mean()
            most_improving_path = improvement.index[improvement.argmax()]
            if improvement[most_improving_path] == 0:
                ended_early = True
                break # perfect already
            portfolio.append(most_improving_path)

        if not ended_early:
            costs.append(all_path_costs[portfolio].min(axis=1).mean())
        infos = [f'+{c / opt - 1:.2%}' for c in costs]
        pf = Portfolio(graph, s, t, portfolio, method=METHOD, infos=infos)
        pf.k = k # in case the portfolio is shorter than k, keep big k as the intention
        pf._impl_hash = HASH
        pf._score = costs[-1]
        pf._opt = opt
        pf._to_cache(cache=cache)
        return pf

if __name__ == "__main__":
    g = Graph.load(city='la')

    # for i, (u, v) in pbar:
    #     pbar.write(f'{u} -> {v}')
    #     for k in range(8, 0, -1):
    #         p = Portfolio.greedy(g, u, v, k=k)
    #     # p.draw(savefig=f'./graph_drawings/sh/{u}-{v}_greedy.png')
    #
    # pbar = tqdm.tqdm(pairs.iterrows())
    # for i, (u, v) in pbar:
    #     pbar.write(f'{u} -> {v}')
    #     for k in range(8, 0, -1):
    #         p = Portfolio.most_frequent(g, u, v, k=k)
    #     # p.draw(savefig=f'./graph_drawings/sh/{u}-{v}_mf.png')
    #
    random.seed(55)
    print('='*50)
    done = set()
    pbar = tqdm.tqdm(range(500))
    for i in pbar:
        u, v = 0, 0
        while (u == v) or (u, v) in done or g.djikstra(u, v) > 1e8:
            u = random.randrange(g.n)
            v = random.randrange(g.n)

        pbar.write(f'{u}-{v}', end=', ' if (i+1) % 10 != 0 else '\n')
        for k in range(8, 0, -1):
            # Portfolio.greedy(g, u, v, k=k, cache='./cache/random_pairs/')
            Portfolio.most_frequent(g, u, v, k=k, cache='./cache/random_pairs/')

        done.add((u, v))

