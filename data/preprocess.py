import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import os
import numpy as np
import geopy.distance

def preprocess_from_adj_n_node_speed(prefix, normalize=False, dir_='data', adj_suffix='_adj', spd_suffix='_speed', ext='csv'):
    node_spd = pd.read_csv(os.path.join(dir_, f'{prefix}{spd_suffix}.{ext}'))
    node_labels = node_spd.columns.to_list()
    n = len(node_labels)
    adj_mat = np.loadtxt(open(os.path.join(dir_, f'{prefix}{adj_suffix}.{ext}')), delimiter=',')
    # print(np.allclose(adj_mat, adj_mat.T))
    # print(np.unique(adj_mat))

    weights = pd.DataFrame() # weight instances
    # reset node columns to 0...n-1 indices
    node_spd.columns = range(node_spd.columns.size)

    a = 0 # arc index
    arcs = []
    for u in range(n):
        
        row = adj_mat[u]
        # columns is Index and has bool indexing, nodes not
        vs = np.arange(n)[row > 0]
        dists = row[row > 0] # binary for shenzhen, not for la though
        for v, d in zip(vs, dists):
            arcs.append((u, v))
            avg_spd = (node_spd[u] + node_spd[v]) * 0.5
            weights[a] = d / avg_spd
            a += 1

    np.savetxt(f'{prefix}_arcs.csv', arcs, fmt='%d', delimiter=',')
    weights.to_csv(f'{prefix}_weights.csv', float_format='%.8f')
    np.savetxt(f'{prefix}_labels_nodes.csv', node_labels, fmt='%s')

def preprocess_shanghai():
    segments = np.loadtxt('sh/segment.csv', delimiter=',', dtype=int)
    arc_nrs = np.loadtxt('sh/selSegs_1.csv', delimiter=',', dtype=int)

    # print(segments.shape)
    # print(arc_nrs.shape)

    arcs = segments[arc_nrs]
    # print(arcs.shape)

    all_nodes = pd.read_csv('sh/node.csv', index_col=0, header=None, names=['lon', 'lat'])
    nodes_ix = np.unique(arcs[:, [1,2]])
    nodes = all_nodes.loc[nodes_ix].copy()

    # print(nodes.head())
    np.savetxt('sh_labels_nodes.csv', nodes_ix, fmt='%s')
    np.savetxt('sh_labels_arcs.csv', arcs[:, 0], fmt='%s')  # basically just a copy of selSegs_1.csv
    nodes.reset_index(drop=True, inplace=True) # we call them 0...n-1
    nodes.to_csv('sh_coords.csv')

    # make arcs use u,v indices instead of labels
    to_indices = lambda col: list(map(lambda x: np.where(nodes_ix == x)[0].item(), arcs[:, col]))
    arcs = np.array(list(zip(to_indices(1), to_indices(2))))
    
    distances = []
    for u, v in arcs:
        u_coords = (nodes.loc[u, 'lat'], nodes.loc[u, 'lon'])
        v_coords = (nodes.loc[v, 'lat'], nodes.loc[v, 'lon'])
        d = geopy.distance.distance(u_coords, v_coords).m
        distances.append(d)
    
    distances = np.array(distances)

    speed = scipy.io.loadmat('sh/selTraffic_1.mat')['selTraffic']
    # weight = pd.DataFrame(1 / speed.T) # calculating time as 1 / speed, since arcs are binary
    weight = pd.DataFrame(distances / speed.T) # calculating time as 1 / speed, since arcs are binary
    # meters / kmh = 1e-3 h, 3.6s
    # speed.plot.hist(column=list(range(20)), bins=100,alpha=0.1)
    # plt.show()
    # print(speed.head())

    np.savetxt('sh_arcs.csv', arcs, fmt='%d', delimiter=',')
    weight.to_csv('sh_weights.csv', float_format='%.8f')

if __name__ == "__main__":
    # preprocess_from_adj_n_node_speed(prefix='sz', dir_='sz')
    # preprocess_from_adj_n_node_speed(prefix='la', dir_='la')
    preprocess_shanghai()
    
