import time
from kmedoids import KMedoids
import itertools
from tqdm import tqdm
from pair_generator import DistanceBucketingPG
from graph import Graph
from portfolio import Portfolio
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

import seaborn as sns
import matplotlib.pyplot as plt

class InducedPartition:
    """a partition of instances induced by p Portfolios,
        usually with all of them being size k
        such that this partition is size at most
        max(k^p, #instances)"""

    # abc
    portfolio_wise_labels = ''.join(
        chr(i) for i in range(ord('a'), ord('z')+1)
    )

    def __init__(self, g: Graph):
        self.p = 0
        self.g = g
        self.instances = pd.DataFrame(index=self.g.weights.index)
        self.inst_vectors = np.zeros((self.g.I, 0))

    def __repr__(self):
        text = f'Partition at {id(self):x} with elements'
        if self.p > 0:
            vc = self.instances.value_counts()
            items = '\n'.join(f"'{''.join(key)}': {count: 5d} #"
                for key, count in vc.items()
            )
            text += '\n' + items
        return text

    def add(self, portfolio: Portfolio):
        c = portfolio.costs()
        winner = np.argmin(c, axis=1)
        letter = np.array(list(InducedPartition.portfolio_wise_labels))[winner]
        self.instances[f'{portfolio.s}-{portfolio.t}'] = letter
        self.inst_vectors = np.hstack([self.inst_vectors, winner.reshape((-1, 1))])
        self.p += 1

    def cluster(self, k=3):
        dists = np.zeros((self.g.I, self.g.I))
        for i in range(self.g.I):
            dists[:, i] = (self.inst_vectors != self.inst_vectors[i]).sum(axis=1)
        kmedoids = KMedoids(n_clusters=k)
        kmedoids.fit_predict(dists)
        return InstancePartition(self.g, kmedoids.labels_)

class InstancePartition:
    def __init__(self, g: Graph, labels: np.ndarray):
        self.g = g
        self.k = np.unique(labels).size
        self.labels = labels

    def __sub__(self, other: InstancePartition):
        assert self.k == other.k
        dists = []
        for perm in itertools.permutations(range(self.k)):
            perm = np.array(list(perm))
            check_against = perm[other.labels]
            dist = (self.labels != check_against).sum()
            dists.append(dist)
        return min(dists)

if __name__ == "__main__":

    # KS = [5, 10, 15, 25, 35, 50, 75]
    KS = []
    N = 20

    g = Graph.load()

    dists = []
    for k in KS:
        parts = []
        print('=' * 20)
        for j in tqdm(range(N), desc=f'k = {k}'):
            indu = InducedPartition(g)
            pg = DistanceBucketingPG(g)

            for _ in tqdm(range(k), leave=False, desc='finding portfolios'):
                u, v = next(pg)
                portfolio = Portfolio.most_frequent(g, u, v, k=3)

                indu.add(portfolio)

            partition = indu.cluster()
            parts.append(partition)

        for i, j in tqdm(itertools.combinations(range(N), 2), desc='dists'):
            dists.append(
                dict(k=k, d=parts[i] - parts[j], i=i, j=j)
            )

    if len(dists):
        dists = pd.DataFrame(dists)
        dists.to_csv('./cache/clustering_dists_temp.csv')

    dists = pd.read_csv('./cache/clustering_dists_temp.csv')
    sns.stripplot(dists, y='d', x='k', native_scale=True, jitter=.2, s=3)
    sns.lineplot(dists, y='d', x='k')
    plt.savefig('./figures/clustering_distances.png')

