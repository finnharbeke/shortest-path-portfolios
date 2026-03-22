import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import plotly.express as px
import plotly

if __name__ == "__main__":
    sns.set()
    
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
    plotly.offline.plot(fig, filename='sh_scatter.html')
    print(dist_uv.head())
