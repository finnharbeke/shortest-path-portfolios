from graph import Graph
from draw import draw_paths
import pandas as pd

if __name__ == "__main__":
    sh = Graph.load()
    
    uvs = pd.read_csv('pick_pairs/sh_picks.csv')
    u, v = uvs.iloc[1]
    shortest = pd.read_csv('pick_pairs/sh_paths.csv', index_col=0, usecols=[0, u*sh.n+v+1])
    print(shortest.head())
    
    freq = shortest.value_counts(normalize=True)
    print(freq)

    k = 9
    top_k = [ix[0] for ix in freq.index[:k]]
    info = [f'{x:.3f}' for x in freq.iloc[:k]]
    draw_paths(sh, top_k, info=info, figsize=(10, 7), savefig='graph_drawings/sh_30-53_greedy.png')

    # print(sh.djikstra(s=u, t=v, path=True))

