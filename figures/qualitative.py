"""
Qualitative comparison of shortest-path portfolios for the Shanghai graph.

Credits:
- _draw_paths, _draw_edge_weights: adapted from draw.py (draw_paths, draw_edge_weights)
- Portfolio computations: adapted from portfolio.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import functools
import seaborn as sns
import os
import sys

import figures
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from graph import Graph
from path import Path
from portfolio import Portfolio

# Colormaps and colors (credited copies from draw.py)
ecm = mpl.color_sequences['Set1']
scm = sns.color_palette('icefire_r', as_cmap=True)

# Human-readable titles for the three methods
TITLES = {
    'greedy': 'Greedy',
    'most frequent shortest paths': 'MFSP',
    'penalised djikstra(1.2)': 'Pen. Dijkstra, $\\lambda=1.2$',
}

def _draw_paths(g, paths, pos, color_nodes=False, info=None, ax=None, title='',
                path_colors=None, **kwargs):
    """
    Credited copy of draw.draw_paths with:
    - color_nodes defaulting to False
    - node positions from shanghai coordinates (pos dict) instead of kamada kawai layout
    - optional path_colors list for custom per-path colouring

    path_colors: list of matplotlib colour specs, one per path in 'paths'.
                 If None, uses the default ecm cycle.
    """
    nxg = nx.MultiDiGraph()
    paths_ab = list(map(functools.partial(Path.to_arc_based, graph=g), paths))
    nxg.add_edges_from(g.arcs)
    arc_to_path = dict()
    for p_ix, p in enumerate(paths_ab):
        for a in Path.to_integers(p):
            arc = g.arcs[a]
            if arc in arc_to_path:
                # hyperarcs
                nxg.add_edge(*arc)
            arc_to_path[arc] = arc_to_path.get(arc, []) + [p_ix]

    edge_color = []
    adj_nodes = set()
    arc_count = dict()
    for (u, v) in list(nxg.edges()):
        if (u, v) not in arc_to_path:
            edge_color.append('#0001')
        else:
            adj_nodes.add(u)
            adj_nodes.add(v)
            ac = arc_count.get((u, v), 0)
            p_ix = arc_to_path[(u, v)][ac]
            if path_colors is not None:
                edge_color.append(mpl.colors.to_hex(path_colors[p_ix]))
            else:
                edge_color.append(mpl.colors.to_hex(ecm[p_ix % len(ecm)]))
            arc_count[(u, v)] = ac + 1

    connectionstyle = [f"arc3,rad={r}" for r in np.linspace(.1, 2, 20)]

    if ax is None:
        ax = plt.figure(figsize=(8, 5)).gca()
    node_color = ['#aaa'] * g.n
    my_kwargs = dict(labels={u: u for u in range(g.n)},
                     font_size=9, node_color=node_color,
                     edge_color=edge_color, connectionstyle=connectionstyle,
                     pos=pos)
    my_kwargs.update(kwargs)
    nx.draw(nxg, ax=ax, **my_kwargs)

    # Tighten axes to node coordinates (ignoring label text which inflates limits)
    xs = [p[0] for key, p in pos.items() if key in adj_nodes]
    ys = [p[1] for key, p in pos.items() if key in adj_nodes]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    pad_x = 0.3 * x_range if x_range > 0 else 0.001
    pad_y = 0.3 * y_range if y_range > 0 else 0.001
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    # write little legend
    if info is not None:
        for p_ix in range(len(paths)):
            c = mpl.colors.to_hex(path_colors[p_ix]) if path_colors is not None else mpl.colors.to_hex(ecm[p_ix % len(ecm)])
            ax.text(0.9, 0.9 - p_ix * 0.06, f'█ {info[p_ix]}',
                    fontdict=dict(color=c),
                    horizontalalignment='right',
                    transform=ax.transAxes)
    if len(title):
        ax.set_title(title, fontsize=10)
    ax.axis('off')

def _draw_edge_weights(g, weights, colorize, path, color, pos, mima=None, ax=None):
    """
    Credited copy of draw.draw_edge_weights with:
    - node positions from shanghai coordinates (pos dict) instead of kamada kawai layout
    - no colorbar drawn in the axis (shared colorbar placed elsewhere)
    """
    nxg = nx.MultiDiGraph()
    nxg.add_edges_from(g.arcs)
    if mima is None:
        ma = max(weights[colorize])
        mi = min(weights[colorize])
    else:
        mi, ma = mima
    norm = mpl.colors.Normalize(mi, ma)
    edge_color = []
    edge_width = []
    adj_nodes = set()
    for (u, v) in list(nxg.edges()):
        a = g.arcs.index((u, v))
        w = weights[a]
        if a not in colorize:
            edge_color.append('#1112')
            edge_width.append(1)
        else:
            adj_nodes.add(u)
            adj_nodes.add(v)
            edge_color.append(mpl.colors.to_hex(scm(norm(w))))
            edge_width.append(2)

    node_color = []
    for u in list(nxg.nodes()):
        if any([u in tup for tup in path]):
            node_color.append(color)
        else:
            node_color.append('#aaa')

    if ax is None:
        ax = plt.figure(figsize=(8, 5)).gca()
    nx.draw(nxg, pos=pos, labels={u: u for u in range(g.n)},
            font_size=3, node_size=10, node_color=node_color,
            edge_color=edge_color, width=edge_width,
            ax=ax)

    # Tighten axes to node coordinates
    xs = [p[0] for key, p in pos.items() if key in adj_nodes]
    ys = [p[1] for key, p in pos.items() if key in adj_nodes]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    pad_x = 0.1 * x_range if x_range > 0 else 0.001
    pad_y = 0.1 * y_range if y_range > 0 else 0.001
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.axis('off')

def _get_path_colours(left_paths, right_paths):
    """
    Build colour lists for left and right portfolios.

    Left paths get ecm[0], ecm[1], ecm[2] (first k colours).
    Right paths that also appear in left reuse the same colour;
    right-only paths get fresh colours starting from index len(left_paths).
    """
    left_colors = [ecm[i % len(ecm)] for i in range(len(left_paths))]
    right_colors = []
    next_idx = len(left_paths)
    for rp in right_paths:
        if rp in left_paths:
            idx = left_paths.index(rp)
            right_colors.append(left_colors[idx])
        else:
            right_colors.append(ecm[next_idx % len(ecm)])
            next_idx += 1
    return left_colors, right_colors


def _winner_counts(portfolio):
    """Return array of counts: how many instances each path wins."""
    costs = portfolio.costs()
    winner = np.argmin(costs.values, axis=1)
    counts = np.bincount(winner, minlength=portfolio.k)
    return counts


def qualitative_comparison(portfolio_left, portfolio_right, node_positions, st):
    """
    Create a side-by-side qualitative comparison figure for two portfolios.

    Uses a 12-column grid (with GridSpec) so that:
      - l spans 5 cols (5/12 width), c spans 2 cols (2/12), r spans 5 cols (5/12)
      - l1..l3, r1..r3 each span 2 columns → each 1/6 width
      - cb spans all 12 cols
    """
    g = portfolio_left.g
    s, t = st
    left_paths = portfolio_left.P
    right_paths = portfolio_right.P
    left_colors, right_colors = _get_path_colours(left_paths, right_paths)

    fig = plt.figure(figsize=(8, 3.5))
    fig.suptitle(f'$({s},{t})$-Portfolios', fontsize=12, y=0.98)

    gs = fig.add_gridspec(
        3, 12,
        height_ratios=[0.6, 0.3, 0.04],
        width_ratios=[1] * 12,  # all equal columns
        hspace=0.15, wspace=0.15
    )

    # Row 0: l (5 cols), c (2 cols), r (5 cols)
    ax_l = fig.add_subplot(gs[0, :5])
    ax_c = fig.add_subplot(gs[0, 5:7])
    ax_r = fig.add_subplot(gs[0, 7:])

    # Row 1: l1-l3, r1-r3 (each 2 cols)
    ax_l1 = fig.add_subplot(gs[1, 0:2])
    ax_l2 = fig.add_subplot(gs[1, 2:4])
    ax_l3 = fig.add_subplot(gs[1, 4:6])
    ax_r1 = fig.add_subplot(gs[1, 6:8])
    ax_r2 = fig.add_subplot(gs[1, 8:10])
    ax_r3 = fig.add_subplot(gs[1, 10:])

    # Row 2: colour bar
    ax_cb = fig.add_subplot(gs[2, :])

    # --- Draw paths on left and right ---
    left_title = (TITLES.get(portfolio_left.method, portfolio_left.method) +
        f': {portfolio_left.score() / portfolio_left._opt:.4f}')
    right_title = (TITLES.get(portfolio_right.method, portfolio_right.method) +
        f': {portfolio_right.score() / portfolio_left._opt:.4f}')

    _draw_paths(g, left_paths, pos=node_positions, ax=ax_l,
                title=left_title,
                path_colors=left_colors, node_size=15, font_size=5)

    _draw_paths(g, right_paths, pos=node_positions, ax=ax_r,
                title=right_title,
                path_colors=right_colors, node_size=15, font_size=5)

    # --- Edge-weight subplots (one per path) ---
    # Compute per-winner-instance mean weights for each path
    left_winner = np.argmin(portfolio_left.costs(), axis=1)
    right_winner = np.argmin(portfolio_right.costs(), axis=1)

    left_arcs_by_path = [Path.to_integers(p) for p in left_paths]
    left_arc_tups_by_path = [[g.arcs[a] for a in p] for p in left_arcs_by_path]
    left_arcs = list(set([
        a for p in left_arcs_by_path for a in p
    ]))
    right_arcs_by_path = [Path.to_integers(p) for p in right_paths]
    right_arc_tups_by_path = [[g.arcs[a] for a in p] for p in right_arcs_by_path]
    right_arcs = list(set([
        a for p in right_arcs_by_path for a in p
    ]))

    # Per-path mean weights (over instances where that path won)
    left_means = []
    right_means = []
    for i in range(3):
        where = g.weights.index[left_winner == i]
        left_means.append((g.weights.loc[where] / g.mean_weights()).mean())
        where = g.weights.index[right_winner == i]
        right_means.append((g.weights.loc[where] / g.mean_weights()).mean())

    all_means = left_means + right_means
    all_arcs_by_subplot = [left_arcs] * 3 + [right_arcs] * 3
    path_arcs_by_subplot = left_arc_tups_by_path + right_arc_tups_by_path

    # Global mima across all per-path means
    global_mi = min([m[left_arcs].min() for m in all_means])
    global_ma = min([m[right_arcs].max() for m in all_means])
    global_mima = (min(global_mi, 2 - global_ma),
                   max(global_ma, 2 - global_mi))

    ew_axes = [ax_l1, ax_l2, ax_l3, ax_r1, ax_r2, ax_r3]
    ew_colors = left_colors + right_colors
    for ax_i, arcs, p, w, color in zip(ew_axes, all_arcs_by_subplot, path_arcs_by_subplot, all_means, ew_colors):
        _draw_edge_weights(g, w, arcs, p, color, pos=node_positions,
                           mima=global_mima, ax=ax_i)
        # Thin coloured border matching the path
        for s in ax_i.spines.values():
            s.set_visible(True)
            s.set_color(mpl.colors.to_hex(color))
            s.set_linewidth(0.8)

    # --- Shared colour bar ---
    norm = mpl.colors.Normalize(*global_mima)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm, scm),
        cax=ax_cb,
        orientation='horizontal',
        ticklocation='bottom'
    )
    cb.set_label('Relative mean edge cost in winning set vs. in the entire distribution')

    # --- Nested pie chart (axis c) ---
    left_counts = _winner_counts(portfolio_left)
    right_counts = _winner_counts(portfolio_right)

    # Outer ring: left portfolio wedges
    outer_sizes = left_counts / left_counts.sum()
    outer_colors = [mpl.colors.to_hex(c) for c in left_colors]

    # Inner ring: right portfolio wedges
    inner_sizes = right_counts / right_counts.sum()
    inner_colors = [mpl.colors.to_hex(c) for c in right_colors]

    # Plot outer ring
    ax_c.pie(outer_sizes, radius=1,
             colors=outer_colors,
             wedgeprops=dict(width=0.3, edgecolor='w'))

    # Plot inner ring (overlaid)
    ax_c.pie(inner_sizes, radius=0.7,
             colors=inner_colors,
             wedgeprops=dict(width=0.3, edgecolor='w'))

    # Label the pie
    # ax_c.text(0, 0, 'winner\nshare', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_c.set_title("Portfolio Path's\nWinning Pr.", fontsize=8)

    return fig


if __name__ == "__main__":
    # Load Shanghai graph
    g = Graph.load('sh', dir_='../data')

    # Load node positions from shanghai coordinates
    cds = pd.read_csv('../data/sh_coords.csv', index_col=0)
    # pos dict: node_id -> (lon, lat) as expected by networkx
    node_pos = {i: (row['lon'], row['lat']) for i, row in cds.iterrows()}

    # Read pick pairs
    pairs = pd.read_csv('../pick_pairs/sh_picks.csv')

    k = 3
    lamb = 1.2

    # Create output directory
    os.makedirs('quali', exist_ok=True)

    for i, (_, row) in enumerate(pairs.iterrows()):
        s, t = int(row['u']), int(row['v'])
        st = (s, t)
        print(f'{s} -> {t}')

        # Compute portfolios
        p_greedy = Portfolio.greedy(g, s, t, k=k, cache="../cache/")
        p_mfsp = Portfolio.most_frequent(g, s, t, k=k, cache="../cache/")
        p_pen  = Portfolio.penalised_djikstra(g, s, t, penalty=lamb, k=k, cache="../cache/")

        # Score all three for fair comparison
        p_greedy.score()
        p_mfsp.score()
        opt = p_greedy._opt if p_greedy._opt is not None else p_mfsp._opt
        # share optimal value so penalised can report factor
        if opt is not None:
            if p_pen._opt is None:
                p_pen._opt = opt
            p_pen._score = p_pen.costs().min(axis=1).mean()

        # Compare each pair
        combos = [
            ('greedy_vs_mfsp',   p_greedy, p_mfsp),
            ('greedy_vs_pen',    p_greedy, p_pen),
            ('mfsp_vs_pen',      p_mfsp,   p_pen),
        ]
        for tag, left, right in combos:
            fig = qualitative_comparison(left, right, node_pos, st)
            savepath = f'quali/sh_{s}_{t}_{tag}.pdf'
            fig.savefig(savepath, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {savepath}')
