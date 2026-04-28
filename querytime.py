import sys
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
from portfolio import Portfolio
import pandas as pd
from graph import Graph



if __name__ == "__main__":
    if not os.path.exists('./cache/sh_querytimes.csv') or len(sys.argv) >= 2:
        g = Graph.load(city='sh')
        pfs = pd.read_csv('./cache/random_pairs/sh_portfolios.csv', index_col=0)
        pfs.drop(pfs.index[~pfs['method'].isin(['greedy', 'most frequent shortest paths'])], inplace=True)

        print(pfs.iloc[0])

        querytimes = pd.DataFrame(columns=['method', 'k', 'cost', 'runtime', 's', 't', 'instance', 'increase'])

        n_pairs = 0
        for i, (s, t) in tqdm.tqdm(pfs[['s', 't']].drop_duplicates().iterrows()):
            st_pfs = pfs[(pfs['s'] == s) & (pfs['t'] == t)]
            for instance in g.weights.sample(n = 30).index:
                start = time.process_time_ns()
                dj_c = g.djikstra(s, t, instance=instance)
                runtime = time.process_time_ns() - start
                querytimes.loc[len(querytimes)] = dict(method = 'djikstra', k = 1,
                                                       cost = dj_c, runtime = runtime,
                                                       s = s, t = t, instance = instance,
                                                       increase = 0)
                for j, row in st_pfs.iterrows():
                    paths = row['portfolio'][1:-1].split(';')
                    portfolio = Portfolio(g, s, t, paths)
                    for k in range(1):
                        start = time.process_time_ns()
                        c = portfolio.compute(instance=instance)
                        runtime = time.process_time_ns() - start
                        querytimes.loc[len(querytimes)] = dict(method = row["method"], k = row["k"],
                                                               cost = c, runtime = runtime,
                                                               s = s, t = t, instance = instance,
                                                               increase = c / dj_c - 1)

            n_pairs += 1
            if n_pairs == 50:
                break

        querytimes.to_csv('./cache/sh_querytimes.csv')
    else:
        querytimes = pd.read_csv('./cache/sh_querytimes.csv', index_col=0)


    plot_df = querytimes
    i = 20
    # plot_df = querytimes[(querytimes['s'] == querytimes.iloc[0]['s']) & (querytimes['t'] == querytimes.iloc[i]['t'])]
    plot_df.sort_values('method', inplace=True, ascending=False)
    gb_qt = plot_df.groupby(by=['method', 'k', 's', 't'], sort=False).agg(dict(cost = 'mean', runtime='mean', method='first', k='first', increase='mean'))
    plot_df = gb_qt
    _ = plt.figure(figsize=(10, 8))
    # for method in plot_df['method'].unique():
    #     sns.regplot(plot_df[plot_df['method'] == method], x='cost', y='runtime', label=method)
    sns.scatterplot(plot_df, x='runtime', y='increase', hue='method', style='k', s=100, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_vs_increase.png')

