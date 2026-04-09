import functools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import folium

from graph import Graph
from path import Path

cm = mpl.colormaps['Spectral']
ecm = mpl.color_sequences['Set1']

def draw_nx(g: Graph, savefig=None, figsize=(8,5)):
    nxg = nx.DiGraph()
    nxg.add_edges_from(g.arcs)
    fig = plt.figure(figsize=figsize)
    node_color = [cm(u / (g.n - 1)) for u in list(nxg)]
    node_color = list(map(mpl.colors.to_hex, node_color))
    nx.draw_kamada_kawai(nxg, labels={u: u for u in range(g.n)}, font_size=9, ax=fig.gca(), node_color=node_color)
    plt.tight_layout()
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, dpi=300)
    plt.close()

def draw_paths(g: Graph, paths, info=None, savefig=None, figsize=(8,5), title=''):
    """ takes paths as list of strings """
    nxg = nx.MultiDiGraph()
    paths = list(map(functools.partial(Path.to_arc_based, graph=g), paths))
    nxg.add_edges_from(g.arcs)
    arc_to_path = dict()
    for p_ix, p in enumerate(paths):
        for a in Path.to_integers(p):
            arc = g.arcs[a]
            if arc in arc_to_path:
                # hyperarcs
                nxg.add_edge(*arc)
            arc_to_path[arc] = arc_to_path.get(arc, []) + [p_ix]
    
    edge_color = []
    arc_count = dict()
    for (u, v) in list(nxg.edges()):
        if (u, v) not in arc_to_path:
            edge_color.append('#0001')
        else:
            ac = arc_count.get((u, v), 0)

            edge_color.append(mpl.colors.to_hex(ecm[arc_to_path[(u,v)][ac] % len(ecm)]))
            arc_count[(u,v)] = ac+1

    connectionstyle = [f"arc3,rad={r}" for r in np.linspace(.1, 2, 20)]

    fig = plt.figure(figsize=figsize)
    node_color = [cm(u / (g.n - 1)) for u in list(nxg)]
    node_color = list(map(mpl.colors.to_hex, node_color))
    nx.draw_kamada_kawai(nxg, labels={u: u for u in range(g.n)}, font_size=9, ax=fig.gca(), node_color=node_color, edge_color=edge_color, connectionstyle=connectionstyle)

    # write little legend
    if info is not None:
        for p_ix in range(len(paths)):
            fig.gca().text(0.9, 0.9 - p_ix * 0.05, f'█: {info[p_ix]}',
                           fontdict=dict(color=mpl.colors.to_hex(ecm[p_ix % len(ecm)])),
                           horizontalalignment='right')
    if len(title):
        plt.title(title)

    plt.tight_layout()
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig, dpi=300)

def draw_coords(g, cds_file):
    cds = pd.read_csv(cds_file, index_col=0)
    m = folium.Map(location=(cds['lat'].mean(), cds['lon'].mean()), zoom_start=15)
    
    for arc in g.arcs:
        u, v = arc
        latlon = lambda row: (row['lat'], row['lon'])
        folium.PolyLine([latlon(0.8*cds.loc[u] + 0.2*cds.loc[v]), latlon(0.2*cds.loc[u] + 0.8*cds.loc[v])], color='#6666a1', weight=5).add_to(m)
        folium.PolyLine([latlon(0.2*cds.loc[u] + 0.8*cds.loc[v]), latlon(cds.loc[v])], color='#c65522', weight=5).add_to(m)

    for i, row in cds.iterrows():
        lat = row['lat']
        lon = row['lon']
        color = mpl.colors.to_hex(cm(i / (len(cds) - 1)))
        folium.CircleMarker(location=(lat, lon), radius=10, tooltip=i, fill=True, fillOpacity=0.5, color=color, fillColor='#fff').add_to(m)
        folium.Marker(location=(lat, lon), icon=folium.features.DivIcon(icon_anchor=(8, 10), html=f'<div style="font-size:11pt; color:#222">{i:02d}</div>')).add_to(m)

    m.save("graph_drawings/shanghai.html")

if __name__ == "__main__":
    sh = Graph.load('sh')

    print(sh)

    print(sh.out_arcs[10])
    print([sh.arcs[a] for a in sh.out_arcs[10]])

    draw_nx(sh, savefig='graph_drawings/sh.png')
    sz = Graph.load('sz')
    draw_nx(sz, savefig='graph_drawings/sz.png', figsize=(9,6))
    la = Graph.load('la')
    draw_nx(la, savefig='graph_drawings/la.png', figsize=(11, 7))

    draw_coords(sh, 'data/sh_coords.csv')
