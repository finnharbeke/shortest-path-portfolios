from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import folium

def draw_nx(g: Graph, savefig=None, figsize=(8,5)):
    nxg = nx.DiGraph()
    nxg.add_edges_from(g.arcs)
    fig = plt.figure(figsize=figsize)
    nx.draw_kamada_kawai(nxg, labels={u: u for u in range(g.n)}, font_size=9, ax=fig.gca())
    plt.tight_layout()
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)

def draw_coords(g, cds_file):
    cds = pd.read_csv(cds_file, index_col=0)
    m = folium.Map(location=(cds['lat'].mean(), cds['lon'].mean()), zoom_start=15)
    
    for arc in g.arcs:
        u, v = arc
        latlon = lambda row: (row['lat'], row['lon'])
        folium.PolyLine([latlon(0.8*cds.loc[u] + 0.2*cds.loc[v]), latlon(0.2*cds.loc[u] + 0.8*cds.loc[v])], color='#666666').add_to(m)
        folium.PolyLine([latlon(0.2*cds.loc[u] + 0.8*cds.loc[v]), latlon(cds.loc[v])], color='#c64444').add_to(m)

    for i, row in cds.iterrows():
        lat = row['lat']
        lon = row['lon']
        folium.CircleMarker(location=(lat, lon), tooltip=i, fill=True).add_to(m)

    m.save("graph_drawings/shanghai.html")

if __name__ == "__main__":
    sh = Graph.load('sh')

    print(sh)

    print(sh.out_arcs[10])
    print([sh.arcs[a] for a in sh.out_arcs[10]])

    # draw_nx(sh, savefig='graph_drawings/sh.png')
    # sz = Graph.load('sz')
    # draw_nx(sz, savefig='graph_drawings/sz.png', figsize=(9,6))
    # la = Graph.load('la')
    # draw_nx(la, savefig='graph_drawings/la.png', figsize=(11, 7))

    draw_coords(sh, 'data/sh_coords.csv')
