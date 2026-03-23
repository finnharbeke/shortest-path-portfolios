import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import plotly.express as px
import plotly

def entropy_mode_freq(col):
    freq = col.value_counts(normalize=True)
    return -(freq * np.log(freq)).sum(), freq.iloc[0]

if __name__ == "__main__":
    sns.set()
    
    # the dist(u,v) random variable, its mean and std
    #################################################
    dists = np.load('sh_dists.npy')

    mean = dists.mean(axis=0)
    std = dists.std(axis=0)

    dist_uv = pd.DataFrame()
    dist_uv['mean'] = mean.reshape((-1,))
    dist_uv['std'] = std.reshape((-1,))
    dist_uv.index = itertools.product(range(dists.shape[-1]), range(dists.shape[-1]))
    u, v = zip(*dist_uv.index)
    dist_uv['u'] = u
    dist_uv['v'] = v
    sns.histplot(dist_uv, x='mean')
    sns.histplot(dist_uv, x='std')
    plt.savefig('sh_hist.png')
    fig = px.scatter(dist_uv, x='mean', y='std', color='u', hover_data=['u', 'v'])
    fig.update_traces(marker=dict(size=10))
    plotly.offline.plot(fig, filename='sh_dist_scatter.html')
    print(dist_uv.head())
    
    # the SP(u, v) random variable, its entropy and the mode's frequency
    ###################################################################### 
    
    paths = pd.read_csv('sh_paths.csv', dtype=str, index_col=0)
    paths.fillna('', inplace=True)
    print(paths.head())
    path_stats = pd.DataFrame(columns=['entropy', 'mode_freq', 'u', 'v'])
    for col_name in paths.columns:
        u, v = [int(x) for x in col_name.split('-')]
        path_stats.loc[col_name] = (*entropy_mode_freq(paths[col_name]), u, v)

    fig = px.scatter(path_stats, x='entropy', y='mode_freq', color='u', hover_data=['u', 'v'])
    fig.update_traces(marker=dict(size=10))
    plotly.offline.plot(fig, filename='sh_path_scatter.html')
