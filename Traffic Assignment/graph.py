import networkx as nx
import numpy as np
from numpy.random import randn
import torch
from helper_functions.save_load import load_obj, save_obj

class Graph():

    def __init__(self, graph_filename: str, sigma=0):

        self.graph = load_obj(graph_filename)
        self.paths = load_obj(graph_filename + '_paths')

        self.source = self.paths[0][0] # Origin
        self.target = self.paths[0][-1] # Destination

        self.len_paths = len(self.paths) # Number of rutes
        self.sigma = sigma # Optimization noise
        self.stochastic = True if sigma > 0 else False

        self.edges = dict()

        # Define edges
        for path in self.paths:
            for node in range(len(path) - 1):
                edge = (path[node], path[node + 1])
                w = self.graph[path[node]][path[node + 1]]['weight']
                self.edges[edge] = {'weight': w, 'load': 0.0}


    def eval(self, x):
        """Evaluates f(x) and grad_f(x).

            f(x) = sum_{e\in E} l_e * c_e(l_e,w_e)
                    where c_e(l_e, w_e) = w_e * (1 + l_e) ^ 2 / 2

            grad_{p}f(x) = sum_{e\in p} z_e(l_e,w_e)
                where z_e(l_e,w_e)  = (l_e * c_e(l_e,w_e))'
                                    = c_e(l_e,w_e) + l_e * c_e'(l_e,w_e)
                                    = c_e(l_e,w_e) + l_e * w_e * (1 + l_e)

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (len_paths, 1).


        """

        x = x.reshape(-1, 1)
        assert x.shape[0] == self.len_paths
        # reset loads
        self.reset_loads()
        # set loads in each path
        self.obj_grad = torch.zeros((self.len_paths, 1), dtype=torch.float64)
        self.set_loads(x)
        # sum all edge loads
        self.obj = torch.zeros((1), dtype=torch.float64)
        for edge in self.edges:
            edge_load = self.edges[edge]['load']
            edge_weight = self.edges[edge]['weight']
            self.obj += edge_load * self.calculate_cost(edge_load, edge_weight)
        # eval gradients
        for p in range(self.len_paths):
            path = self.paths[p]
            for node in range(len(path) - 1):
                # Get edges tuple (key)
                edge = (path[node], path[node + 1])
                edge_load = self.edges[edge]['load'] # Z_e
                edge_weight = self.edges[edge]['weight'] # W_e
                # (df)_p
                self.obj_grad[p] += self.calculate_cost(edge_load, edge_weight) + \
                            edge_load * self.calculate_dcost(edge_load, edge_weight)


    def calculate_cost(self, load, weight):
        '''
        # cost_e = 0.5 * W_e (1 + l_e) ** 2 + sigma * N(0, 1)
        '''
        cost = (weight * (1 + load) ** 2) / 2
        return cost + randn(1) * self.sigma if self.stochastic else cost

    def calculate_dcost(self, load, weight):
        '''
        # dcost_e = W_e * (1 + l_e) + sigma * N(0, 1)
        '''
        dcost = weight * (1 + load)
        return dcost + randn(1) * self.sigma if self.stochastic else dcost


    def reset_loads(self):
        '''
        Sets loads to 0.
        '''
        for edge in self.edges:
            self.edges[edge]['load'] = 0.0


    def set_loads(self, x):
        '''
        Sets loads for each edge.
        '''
        for p in range(self.len_paths):
            path = self.paths[p]
            for node in range(len(path) - 1):
                # Get edges tuple (key)
                edge = (path[node], path[node + 1])
                self.edges[edge]['load'] += x[p]




if __name__ == '__main__':
    path = "./saved_items/graph_N50_cutoff6"
    g = Graph(path, 0.2)
    np.random.seed(10)
    x = np.random.randn((g.len_paths))
    x = torch.tensor(x)
    # print(x.shape)
    g.eval(x)
    print(g.obj_grad)
    print(g.obj)
