import sys
sys.path.insert(0, '../')
from helper_functions.save_load import save_obj

import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch


class Geo_graph_3d():

    def __init__(self, n_nodes, radius, seed=None, pos=None):
        self.n_nodes = n_nodes
        self.radius = radius
        if seed is not None:
            random.seed(seed)

        pos = {i: (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n_nodes)}

        self.graph = nx.random_geometric_graph(n_nodes, radius=radius, dim=3, pos=pos)


        self.size = self.graph.size()
        self.edges = self.graph.edges
        self.Adj = self.get_adj()
        self.degs = np.array([self.graph.degree(i) for i in range(self.n_nodes)]).astype('int')
        self.pos = self.get_pos()
        self.pos_array = self.get_pos_array()

    def get_adj(self):

        Adj = np.zeros((self.n_nodes, self.n_nodes))
        for edge in self.graph.edges:
            node1 = edge[0]
            node2 = edge[1]
            Adj[node1, node2] = 1
        Adj = Adj + Adj.T
        return Adj.astype('int')

    def get_pos(self):

        return nx.get_node_attributes(self.graph, 'pos')

    def get_pos_array(self):
        dim = 3
        pos = np.zeros((self.n_nodes, dim))
        for node in self.pos:
            pos[node] = self.pos[node]
        return pos

    def plot_graph_3D(self, angle=0):
        # Get node positions
        pos = self.pos

        # Get number of nodes
        n = self.n_nodes

        # Get the maximum number of edges adjacent to a single node
        edge_max = self.degs.max().item()

        # Define color range proportional to number of edges adjacent to a single node
        colors = [plt.cm.plasma(self.degs[i].item()/edge_max) for i in range(self.n_nodes)]

        # 3D network plot
        with plt.style.context(('seaborn-bright')):

            fig = plt.figure(figsize=(10,7))
            ax = Axes3D(fig)

            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            for key, value in self.pos.items():
                xi = value[0]
                yi = value[1]
                zi = value[2]

                # Scatter plot
                ax.scatter(xi, yi, zi, color=colors[key], s=10 + 20 * self.degs[key], edgecolors='k', alpha=0.7)

            # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            for i,j in enumerate(self.edges):

                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
                ax.plot(x, y, z, c='black', alpha=0.5)

        # Set the initial view
        ax.view_init(40, angle)

        # Hide the axes
        ax.set_axis_off()


        plt.savefig("./figures/3dgraph_N_{}_radius_{}.png".format(self.n_nodes, self.radius), dpi=500)
        plt.show()

        return

if __name__ == '__main__':
    G = Geo_graph_3d(n_nodes=100, radius=0.6, seed=1)
    print(G.Adj.sum(1), G.Adj.sum(0))
    print(G.degs)
    print(G.Adj)
    print(G.pos[19])
    print(G.pos_array[19])
    G.plot_graph_3D()
    # print(G.pos)
    # network_plot_3D(G, angle=1)
    # print(len(list(G.edges)), G.size(), nx.get_node_attributes(G, 'pos'))
