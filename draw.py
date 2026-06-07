import functools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import folium
from dotenv import load_dotenv
import os
from folium.plugins import MiniMap

load_dotenv()

from graph import Graph
from path import Path

cm = mpl.colormaps['Spectral']
ecm = mpl.color_sequences['Set1']
ecm2 = mpl.color_sequences['Set2']
scm = sns.color_palette('flare', as_cmap=True)

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

def draw_edge_weights(g: Graph, weights, colorize, st, savefig=None, mima=None, ax=None, figsize=(8, 5)):
    nxg = nx.MultiDiGraph()
    nxg.add_edges_from(g.arcs)
    if mima is None:
        ma, mi = max(weights[colorize]), min(weights[colorize])
    else:
        mi, ma = mima
    norm = mpl.colors.Normalize(mi, ma)
    edge_color = []
    edge_width = []
    for (u, v) in list(nxg.edges()):
        a = g.arcs.index((u, v))
        w = weights[a]
        if a not in colorize:
            edge_color.append('#1112')
            edge_width.append(1)
        else:
            edge_color.append(
                mpl.colors.to_hex(scm(norm(w)))
            )
            edge_width.append(3)
    node_color = []
    for u in nxg.nodes():
        if u == st[0]:
            node_color.append(ecm2[0])
        elif u == st[1]:
            node_color.append(ecm2[1])
        else:
            node_color.append('#aaa')

    if ax is None:
        ax = plt.figure(figsize=figsize).gca()
    nx.draw_kamada_kawai(nxg, labels={u: u for u in range(g.n)},
                         font_size=6, node_size=100, node_color=node_color,
                         edge_color=edge_color, width=edge_width,
                         ax=ax)
    plt.colorbar(mpl.cm.ScalarMappable(norm, scm), ax=ax,
                 location='bottom', pad=0, fraction=0.05, aspect=80)
    if savefig is not None:
        plt.savefig(savefig, dpi=300)

def draw_paths(g: Graph, paths, color_nodes=True, info=None, ax=None, savefig=None, figsize=(8,5), title='', **kwargs):
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

    if ax is None:
        ax = plt.figure(figsize=figsize).gca()
    node_color = [cm(u / (g.n - 1)) for u in list(nxg)] if color_nodes else ['#aaa'] * g.n
    node_color = list(map(mpl.colors.to_hex, node_color))
    my_kwargs = dict(labels={u: u for u in range(g.n)},
                     font_size=9, node_color=node_color,
                     edge_color=edge_color, connectionstyle=connectionstyle)
    my_kwargs.update(kwargs)
    nx.draw_kamada_kawai(nxg, ax=ax, **my_kwargs)

    # write little legend
    if info is not None:
        for p_ix in range(len(paths)):
            ax.text(0.9, 0.9 - p_ix * 0.06, f'█ {info[p_ix]}',
                           fontdict=dict(color=mpl.colors.to_hex(ecm[p_ix % len(ecm)])),
                           horizontalalignment='right')
    if len(title):
        ax.set_title(title)

    if savefig is not None:
        plt.savefig(savefig, dpi=300)

def draw_partition(ew_kwargs, p_kwargs, bar_kwargs, savefig, title):
    ### plot layout, left big, edge weights, top right paths, bottom right bars of path costs

    fig, axes = plt.subplot_mosaic([['a', 'b'], ['a', 'c']], width_ratios=[2, 1])
    draw_edge_weights(**ew_kwargs, ax=axes['a'])
    draw_paths(**p_kwargs, ax=axes['b'], color_nodes=False)
    paths = range(len(bar_kwargs['iP']))
    costs = [bar_kwargs['weights'][p].sum() for p in bar_kwargs['iP']]
    sns.barplot(x=paths, y=costs, hue=paths, palette=ecm, ax=axes['c'])
    plt.title(title)
    fig.tight_layout()
    fig.savefig(savefig, dpi=300)

from draw_helpers import _arrowhead, _shorten_line

def draw_coords(g, cds_file):
    cds = pd.read_csv(cds_file, index_col=0)
    # t = folium.raster_layers.TileLayer(tiles='OpenStreetMap', attr='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', referrerPolicy='strict-origin-when-cross-origin')
    t = folium.raster_layers.TileLayer(tiles=f'https://api.thunderforest.com/mobile-atlas/{{z}}/{{x}}/{{y}}{{r}}.png?apikey={os.getenv("THUNDERFOREST_APIKEY")}', 
	attr= '&copy; <a href="http://www.thunderforest.com/">Thunderforest</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    apikey=os.getenv("THUNDERFOEST_APIKEY"),
	maxZoom= 22
    )
    m = folium.Map(location=(cds['lat'].quantile(.35), cds['lon'].mean()), zoom_start=16, tiles=t, control_scale=True)
    fontsize = 16

    for arc in g.arcs:
        u, v = arc
        latlon = lambda row: (row['lat'], row['lon'])

        # deepseek-v4-flash
        # CircleMarker radius = fontsize * 1.2 = 24 px.
        # At zoom_start=16, Shanghai (~31°N): ~2 m/px, so 24 px ≈ 48 m.
        # 48 m ≈ 0.00043° lat
        # finn using .0006
        node_r_deg = 0.0006
        p1, p2 = _shorten_line(latlon(cds.loc[u]), latlon(cds.loc[v]), node_r_deg)
        folium.PolyLine(_arrowhead([p1, p2]), color='#222', weight=fontsize * .2).add_to(m)

    for i, row in cds.iterrows():
        lat = row['lat']
        lon = row['lon']
        color = mpl.colors.to_hex(cm(i / (len(cds) - 1)))
        folium.CircleMarker(location=(lat, lon), radius=fontsize * 1.2, weight=fontsize * .33, tooltip=str(i), fill=True, fillOpacity=0.5, color=color, fillColor='#fff').add_to(m)
        folium.Marker(location=(lat, lon),
                      icon=folium.features.DivIcon(icon_anchor=(round(fontsize * .75), fontsize),
                                                   html=f'<div style="font-size:{fontsize}pt; color:#222; font-weight: bold">{i:02d}</div>'
                                                   )
                      ).add_to(m)

    mini = MiniMap(tile_layer=t, width=500, height=300)
    mini.add_to(m)
    m.save("docs/shanghai.html")

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
