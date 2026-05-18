from tqdm import tqdm
from pair_generator import DistanceBucketingPG
from graph import Graph
from portfolio import Portfolio
import pandas as pd
import numpy as np

class Partition:
    """a partition of instances given p Portfolios,
        usually with all of them being size k
        such that this partition is size k^p"""

    # abc
    portfolio_wise_labels = ''.join(
        chr(i) for i in range(ord('a'), ord('z')+1)
    )

    def __init__(self, g: Graph):
        self.p = 0
        self.g = g
        self.instances = pd.DataFrame(index=self.g.weights.index)
        self.instances['partition'] = ''

    def __repr__(self):
        vc = self.instances['partition'].value_counts()
        items = f'\n'.join(f"'{key}': {count: 5d} #"
            for key, count in vc.items()
        )
        return f'Partition at {id(self):x} with elements\n{items}'

    def add(self, portfolio: Portfolio):
        c = portfolio.costs()
        winner = np.argmin(c, axis=1)
        letter = np.array(list(Partition.portfolio_wise_labels))[winner]
        self.instances['partition'] += letter

if __name__ == "__main__":
    g = Graph.load()
    part = Partition(g)
    print(part)

    sampler = DistanceBucketingPG(g)
    for _ in tqdm(range(5)):
        u, v = next(sampler)
        pf = Portfolio.most_frequent(g, u, v, k=3)
        part.add(pf)
        print(part)

