import time
import random
import abc
from graph import Graph
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class PairGenerator(metaclass=abc.ABCMeta):
    def __init__(self, g: Graph):
        self.g = g

    @abc.abstractmethod
    def __next__(self) -> tuple[int, int]:
        pass

class UniformPG(PairGenerator):
    def __next__(self):
        d = np.inf
        while np.isinf(d):
            u = random.randrange(self.g.n)
            v = random.randrange(self.g.n)
            d = self.g.djikstra(u, v)
        return (u, v)

class UniformDistinctPG(PairGenerator):
    def __init__(self, g: Graph):
        super().__init__(g)
        self.sub_generator = UniformPG(self.g)

    def __next__(self):
        u, v = 0, 0
        while u == v:
            u, v = next(self.sub_generator)
        return (u, v)

class DistanceBucketingPG(PairGenerator):
    def __init__(self, g: Graph, k=5, infer_from_n=100):
        super().__init__(g)
        self.k = k
        udpg = UniformDistinctPG(self.g)
        max_d = 0
        min_d = 1e9
        for _ in range(infer_from_n):
            u, v = next(udpg)
            d = self.g.djikstra(u, v)
            max_d = max(max_d, d)
            min_d = min(min_d, d)

        stretch = max_d - min_d
        interval = stretch / k
        self.cuts = []
        for j in range(1, k):
            self.cuts.append(min_d + j * interval)

        self.sub_generator = UniformDistinctPG(self.g)
        self.buckets_used = set()

    def bucket(self, d):
        bucket = 0
        while bucket < self.k - 1 and d >= self.cuts[bucket]:
            bucket += 1
        return bucket

    def __next__(self):
        " return 1 pair per bucket until all buckets returned one, repeat "
        b = -1
        while b < 0 or b in self.buckets_used:
            u, v = next(self.sub_generator)
            d = self.g.djikstra(u, v)
            b = self.bucket(d)
        self.buckets_used.add(b)
        if len(self.buckets_used) == self.k:
            self.buckets_used.clear()
        return u, v

if __name__ == "__main__":
    g = Graph.load()
    un = UniformPG(g)
    ud = UniformDistinctPG(g)
    start = time.time()
    db = DistanceBucketingPG(g, 5)
    print(f'bucketing init took {time.time() - start:.2f}s')

    df = pd.DataFrame(columns=['distance', 'generator'])
    for _ in tqdm(range(1000)):
        u, v = next(ud)
        df.loc[len(df)] = (g.djikstra(u, v), 'uniform')
        u, v = next(db)
        df.loc[len(df)] = (g.djikstra(u, v), 'bucketing')

    bins = []
    step = db.cuts[1] - db.cuts[0]
    min_ = db.cuts[0] - step
    step /= 3
    for j in range(db.k * 3 + 1):
        bins.append(min_ + j * step)
    sns.histplot(df, x='distance', hue='generator', bins=bins, multiple='dodge')
    for c in db.cuts:
        plt.axvline(c)
    plt.savefig('sampling_behaviour.png')
