import draw
from portfolio import Portfolio
from graph import Graph
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    k = 3

    pfs = pd.read_csv('./cache/random_pairs/sh_portfolios.csv', index_col = 0)
    pfs.drop(pfs.index[~(pfs['k'] == 3)], inplace=True)
    pfs.drop(pfs.index[~pfs['method'].isin(['greedy', 'most frequent shortest paths'])], inplace=True)

    mf_vs_greedy = pd.DataFrame(columns=['greedy', 'mfsp', 's', 't'])
    for (s, t), group in pfs.groupby(['s', 't']):
        # print(group)
        greedy = group[(group['method'] == 'greedy')].iloc[0]
        mfsp = group[~(group['method'] == 'greedy')].iloc[0]
        mf_vs_greedy.loc[len(mf_vs_greedy)] = dict(greedy = greedy['factor'], mfsp = mfsp['factor'], s=s, t=t)

    mf_vs_greedy['diff'] = mf_vs_greedy['mfsp'] - mf_vs_greedy['greedy'] 

    print(mf_vs_greedy[mf_vs_greedy['diff'] > 0.004])

    ix = 78
    print(mf_vs_greedy.loc[ix])
    s, t = mf_vs_greedy.loc[ix, ['s', 't']]
    relevant = pfs[(pfs['s'] == s) & (pfs['t'] == t)].sort_values('method')
    gr_row, mf_row = relevant.iloc[0], relevant.iloc[1]
    
    g = Graph.load('sh')
    greedy = Portfolio(g, s, t, paths=gr_row['portfolio'][1:-1].split(';'), infos=gr_row['infos'][1:-1].split(';;'))
    mfsp = Portfolio(g, s, t, paths=mf_row['portfolio'][1:-1].split(';'), infos=mf_row['infos'][1:-1].split(';;'))

    draw.draw_edge_weights(g, g.weights.mean(), [a for p in greedy.iP for a in p], './graph_drawings/sh/part.png')

    winner = np.argmin(greedy.costs(), axis=1)
    for i in range(greedy.k):
        where = g.weights.index[winner == i]
        w = g.weights.loc[where]
        draw.draw_edge_weights(g, w.mean(), [a for p in greedy.iP for a in p], f'./graph_drawings/sh/part{i}.png')


    # todo: partition instances and draw edge weights for each mean

