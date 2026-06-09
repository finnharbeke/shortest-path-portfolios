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
    plot_df['approx'] = plot_df['increase'] + 1
    plot_df['st'] = plot_df['s'].astype(str) + ' - ' + plot_df['t'].astype(str)
    gb_qt = plot_df.groupby(by=['method', 'k', 'st'], sort=False).agg(dict(cost = 'mean', runtime='mean', method='first', k='first', approx='mean', st='first'))
    plot_df = gb_qt
    not_easy = plot_df['st'][(plot_df['k'] == 3) & (plot_df['approx'] != 1)]
    plot_df = plot_df[plot_df['st'].isin(not_easy)]
    top_10= plot_df['st'].value_counts().head(10).index
    plot_df = plot_df[plot_df['st'].isin(top_10)]
    _ = plt.figure(figsize=(10, 8))
    # for method in plot_df['method'].unique():
    #     sns.regplot(plot_df[plot_df['method'] == method], x='cost', y='runtime', label=method)
    proper = {'greedy': 'Greedy', 'most frequent shortest paths': 'MFSP', 'djikstra': 'Dijkstra'}
    plot_df['method'] = plot_df['method'].str.replace(proper)
    markers = {'Greedy': 'o', 'MFSP': 's', 'Dijkstra': 'X'}
    ax = sns.scatterplot(plot_df, x='runtime', y='approx', hue='st', style='method', s=100, alpha=0.7, markers=markers)
    handles, labels = ax.get_legend_handles_labels()
    # Create a single handle (e.g., first one)
    custom_handle = [handles[1]]  # or create a custom Patch
    sty_h, sty_l = zip(*list(filter(lambda xhl: not xhl[1][0].isdigit() and (xhl[1] not in ['st', 'method']), zip(handles, labels))))
    ax.legend(handles=custom_handle + list(sty_h), labels=['s - t'] + list(sty_l))
    ax.set_xlabel(r'$T^q(ALG)$ (ns)')
    ax.set_ylabel(r'$\overline{\alpha} (\text{ALG}, (G, \mathcal{C}, \mathcal{D}); s, t)$')

    plt.tight_layout()
    plt.savefig('figures/runtime_vs_approx.png')

