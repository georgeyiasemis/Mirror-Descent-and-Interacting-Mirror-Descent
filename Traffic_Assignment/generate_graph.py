import sys
sys.path.insert(0, '../')
import networkx as nx
from networkx import geographical_threshold_graph as geo_thresh
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import torch
from helper_functions.save_load import save_obj


def gen_geo(num_nodes, theta, lambd, source, target, cutoff, seed=None):
    """Generates a random graph with threshold theta consisting of 'num_nodes'
        and paths with maximum length 'cutoff' between 'source' adn target.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    theta : float
        Threshold of graph.
    lambd : float
        Weights of graph are generated randomly from exp(lambd) distribution.
    source : int
        Origin of path. Must be in range(0, num_nodes).
    target : int
        Destination of path. Must be in range(0, num_nodes).
    cutoff : int
        Maximum path length.
    seed : int
        Set random seed if not None.

    Returns
    -------
    object of type graph
        Generated graph.

    """


    file_name = './saved_items/graph_N' + str(num_nodes) + '_cutoff' + str(cutoff)
    if seed != None:
        np.random.seed(seed)
        rand.seed(seed)
        torch.manual_seed(seed)

    weights = { node: rand.expovariate(lambd) for node in range(num_nodes)}

    graph = geo_thresh(num_nodes, theta, weight=weights)

    for (ni, nj) in graph.edges():
        graph.edges[ni,nj]['weight'] = weights[ni] + weights[nj]

    plt.figure(figsize=(10,5))
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.savefig('./figures/graph_N' + str(num_nodes) + str(".png"), dpi=500)
    plt.show()
    save_obj(graph, file_name)
    paths = nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff)
    paths = list(paths)
    save_obj(paths, file_name + '_paths')

    print('Paths length: ', len(paths))
    return graph


if __name__ == '__main__':

    # ARGS = [{'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 12,
    #                 'target': 37, 'cutoff': 4, 'seed': 10 },
    #         {'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 12,
    #                 'target': 37, 'cutoff': 5, 'seed': 10 },
    #         {'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 14,
    #                 'target': 37, 'cutoff': 6, 'seed': 10 },
    #         {'num_nodes': 100, 'theta': 16, 'lambd': 4, 'source': 74,
    #                 'target': 5, 'cutoff': 6, 'seed': 10 },
    #         {'num_nodes': 100, 'theta': 16, 'lambd': 4, 'source': 74,
    #                 'target': 5, 'cutoff': 7, 'seed': 10 }]
    ARGS = [{'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 46,
                    'target': 32, 'cutoff': 4, 'seed': 0 },
            {'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 46,
                    'target': 32, 'cutoff': 5, 'seed': 0 },
            {'num_nodes': 50, 'theta': 16, 'lambd': 1.2, 'source': 46,
                    'target': 32, 'cutoff': 6, 'seed': 0 },
            {'num_nodes': 100, 'theta': 16, 'lambd': 4, 'source': 94,
                    'target': 77, 'cutoff': 6, 'seed': 100 },
            {'num_nodes': 100, 'theta': 16, 'lambd': 4, 'source': 94,
                    'target': 77, 'cutoff': 7, 'seed': 100 }]

    for args in ARGS:
        gen_geo(**args)
