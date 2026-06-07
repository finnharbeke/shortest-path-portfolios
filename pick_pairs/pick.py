import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import plotly.express as px
import plotly.graph_objects as go
import plotly

def uv_scatterplot(x, y, filename):
    fig1 = px.scatter(path_stats, x=x, y=y, color='t', hover_data=['s', 't'])
    fig1.update_traces(marker=dict(size=10))
    fig2 = px.scatter(path_stats, x=x, y=y, color='s', hover_data=['s', 't'])
    fig2.update_traces(marker=dict(size=8, symbol='star-diamond'))
    fig = go.Figure(data = fig1.data + fig2.data)
    fig.update_layout(colorscale=dict(sequential=px.colors.diverging.Spectral), xaxis_title=x, yaxis_title=y)
    plotly.offline.plot(fig, filename=filename)

def entropy_mode_freq(col):
    freq = col.value_counts(normalize=True)
    return -(freq * np.log(freq)).sum(), freq.iloc[0]

if __name__ == "__main__":
    sns.set()

    # the dist(u,v) random variable, its mean and std
    #################################################
    dists = np.load('sh_dists.npy')
    dists = dists.reshape((dists.shape[0], -1))

    mean_dist = dists.mean(axis=0)
    std = dists.std(axis=0)

    path_stats = pd.DataFrame()
    path_stats['expected s-t distance'] = mean_dist.reshape((-1,))
    path_stats['standard deviation of s-t distance'] = std.reshape((-1,))
    # shuts down python v fast
    n = int(np.sqrt(dists.shape[-1]))
    print(n)
    # u, v = zip(*itertools.product(range(dists.shape[-1]), range(dists.shape[-1])))
    # n is root of dists.shape -1, not that itself
    path_stats['s'] = np.repeat(np.arange(n), n)
    path_stats['t'] = np.tile(np.arange(n), n)
    path_stats.set_index(['s', 't'], inplace=True, drop=False)
    sns.histplot(path_stats, x='expected s-t distance')
    sns.histplot(path_stats, x='standard deviation of s-t distance')
    plt.savefig('sh_hist.png')

    uv_scatterplot('expected s-t distance', 'standard deviation of s-t distance', 'sh_dist_scatter.html')
    print(path_stats.head())
    
    # the SP(u, v) random variable, its entropy and the mode's frequency
    ###################################################################### 
    
    paths = pd.read_csv('sh_paths.csv', dtype=str, index_col=0)
    paths.fillna('', inplace=True)
    print(paths.head())
    path_stats['entropy'] = pd.Series(dtype=float)
    path_stats['frequency of mode'] = pd.Series(dtype=float)

    for col_name in paths.columns:
        u, v = [int(x) for x in col_name.split('-')]
        e, a = entropy_mode_freq(paths[col_name])
        path_stats.at[(u, v), 'entropy'] = e
        path_stats.at[(u, v), 'frequency of mode'] = a

    uv_scatterplot('entropy', 'frequency of mode', 'sh_path_scatter.html')
    print(path_stats.head())

    # SP(u, v)'s entropy vs. dist(u, v) STD
    ################################################## 

    uv_scatterplot('entropy', 'standard deviation of s-t distance', 'sh_entr_std_scatter.html')    
    
    path_stats.to_csv('sh_path_stats.csv')
