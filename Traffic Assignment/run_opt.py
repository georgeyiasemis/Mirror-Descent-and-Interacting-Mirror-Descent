import os
import numpy as np
import torch
from helper_functions.projections import projection_simplex
from helper_functions.save_load import save_obj
from graph import Graph
from tqdm import trange

def run_GD(graph_filename: str, path_filename: str, sigma: float,
            lr: float,  max_iters: int, Niid: int, decreasing_lr=False, seed=None):

    """Runs Gradient Descent Optimisation for the Traffic Assignment Problem
        (Equation (5.1) of the project)

    Parameters
    ----------
    graph_filename : str
        Path for Graph object.
    path_filename : str
        Path for list of all (simple) paths to optimise.
    sigma : float
        Noise parameter. If sigma > 0 then optimisation is Stochastic.
    lr : float
        Learning Rate.
    max_iters : int
        Number of iterations to run optimisation.
    Niid : int
        Number of iid copies.
    decreasing_lr : type
        If true the learning rate is decreasing.
    seed : int
        Initial random seed.

    """

    assert sigma >= 0.0, "Sigma should be non negative."

    iters = [0 for i in range(max_iters)]
    lr_const = 1
    graph = Graph(graph_filename, sigma)
    len_paths = graph.len_paths
    print(len_paths)
    print('GD')
    print('--')
    print('Number of iid runs: ', Niid)
    print('sigma = ', sigma)

    with trange(Niid, ncols=150) as tqdm:
        for i in tqdm:
            if sigma > 0:
                tqdm.set_description('SGD')
            else:
                tqdm.set_description('GD')

            x_t = torch.ones((len_paths, 1), dtype=torch.float64) / len_paths
            graph.eval(x_t)

            iters[0] += graph.obj.item() / Niid

            for t in range(1, max_iters):
                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()

                # x_{t+1} = x_t - h_t * e * grad_f(x_t)
                x_t = x_t.reshape(-1, 1) - lr * lr_const * graph.obj_grad.reshape(-1, 1)

                # x_{t+1} = Π_X(x_{t+1})
                x_t = projection_simplex(x_t)

                graph.eval(x_t)

                iters[t] += graph.obj.item() / Niid

            tqdm.set_postfix(Loss=iters[-1])

    return iters

def run_MD(graph_filename: str, path_filename: str, sigma: float,
            lr: float,  max_iters: int, Niid: int, decreasing_lr=False, seed=None):

    """Runs Mirror Descent Optimisation for the Traffic Assignment Problem
        (Equation (5.1) of the project)

    Parameters
    ----------
    graph_filename : str
        Path for Graph object.
    path_filename : str
        Path for list of all (simple) paths to optimise.
    sigma : float
        Noise parameter. If sigma > 0 then optimisation is Stochastic.
    lr : float
        Learning Rate.
    max_iters : int
        Number of iterations to run optimisation.
    Niid : int
        Number of iid copies.
    decreasing_lr : type
        If true the learning rate is decreasing.
    seed : int
        Initial random seed.

    """

    assert sigma >= 0.0, "Sigma should be non negative."

    graph = Graph(graph_filename, sigma)
    len_paths = graph.len_paths

    print('MD')
    print('--')
    print('Number of iid runs: ', Niid)
    print('sigma = ', sigma)

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1

    with trange(Niid, ncols=150) as tqdm:
        for i in tqdm:
            if sigma > 0:
                tqdm.set_description('SMD')
                x_t = torch.rand(len_paths, 1, dtype=torch.float64)
                x_t = x_t / torch.sum(x_t)
            else:
                tqdm.set_description('MD')
                x_t = torch.ones((len_paths, 1), dtype=torch.float64) / len_paths

            graph.eval(x_t)

            iters[0] += graph.obj.item() / Niid

            for t in range(1, max_iters):
                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()

                x_t = x_t.reshape(-1, 1) * torch.exp(- lr * lr_const * graph.obj_grad).reshape(-1, 1)

                x_t = x_t / torch.sum(x_t)

                graph.eval(x_t)

                iters[t] += graph.obj.item() / Niid

            tqdm.set_postfix(Loss=iters[-1])
    return iters

def run_IMD(graph_filename: str, path_filename: str, sigma: float,
            lr: float,  max_iters: int, num_particles: int, eps=1, decreasing_lr=False, seed=None):

    """Runs Interacting Mirror Descent Optimisation for the Traffic Assignment Problem
        (Equation (5.1) of the project)

    Parameters
    ----------
    graph_filename : str
        Path for Graph object.
    path_filename : str
        Path for list of all (simple) paths to optimise.
    sigma : float
        Noise parameter. If sigma > 0 then optimisation is Stochastic.
    lr : float
        Learning Rate.
    max_iters : int
        Number of iterations to run optimisation.
    Niid : int
        Number of iid copies.
    decreasing_lr : type
        If true the learning rate is decreasing.
    seed : int
        Initial random seed.

    """

    assert sigma >= 0.0, "Sigma should be non negative."
    assert num_particles > 1

    graph = Graph(graph_filename, sigma)
    len_paths = graph.len_paths
    print(len_paths)
    print('InterMD')
    print('-------')
    print('Number of iteracting paricles: ', num_particles)
    print('sigma = ', sigma)

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1

    particles = torch.rand(len_paths, num_particles, dtype=torch.float64)
    particles = particles / particles.sum(0)

    interaction_mat = torch.ones((num_particles, num_particles),
                                dtype=torch.float64) / num_particles

    obj = float('inf')

    with trange(max_iters, ncols=150) as tqdm:
        for t in tqdm:
            if sigma > 0:
                tqdm.set_description('SIMD')
            else:
                tqdm.set_description('IMD')
            obj_min = float('inf')
            obj_avg = 0.0
            A = particles

            for p in range(num_particles):
                x_p = particles[:, p]
                graph.eval(x_p)
                obj_avg += graph.obj.item() / num_particles

                obj_min = graph.obj.item() if obj_min > graph.obj.item() else obj_min

                # part_mean = (particles * interaction_mat[p, :]).sum(1) / interaction_mat[p, :].sum()
                #
                # part_mean += particles[:, p] * (1 - interaction_mat[p, p])

                # x_p = part_mean * torch.exp(- lr * graph.obj_grad).reshape(-1)
                x_p = torch.prod(torch.pow(particles, interaction_mat[p]), axis=1)

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t + 1).item()

                x_p = x_p * torch.exp(- lr * lr_const * eps * graph.obj_grad).reshape(-1)

                x_p /= torch.sum(x_p)

                A[:, p] = x_p

            iters[t] = obj_avg

            tqdm.set_postfix(Loss=iters[t])

    return iters

if __name__ == "__main__":
    path = './saved_items/50Nodes/'
    graph_filename = path + 'graph_N50_cutoff5'
    path_filename = graph_filename + 'paths'


    sigma = 0.0
    lr = 0.01
    lr_md = 0.2
    Niid = 5
    max_iters = 500
    num_particles = 5
    decreasing_lr = False
    eps = 1
    seed = 0

    path = path + 'Results/' + graph_filename[-7:]
    if not os.path.exists(path):
        os.mkdir(path)

    args_gd = {'graph_filename': graph_filename, 'path_filename': path_filename, 'sigma': sigma,
     'lr': lr, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_md = {'graph_filename': graph_filename, 'path_filename': path_filename, 'sigma': sigma,
     'lr': lr_md, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_imd = {'graph_filename': graph_filename, 'path_filename': path_filename, 'sigma': sigma,
     'lr': lr_md, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}


    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_' + str(num_particles) + '_iters_' + str(max_iters))
    losses_gd = run_GD(**args_gd)
    # save_obj((losses_gd, args_gd), path + '/GD_Niid_' + str(Niid) + '_iters_' + str(max_iters))
    losses_md = run_MD(**args_md)
    # save_obj((losses_md, args_md), path + '/MD_Niid_' + str(Niid) + '_iters_' + str(max_iters))


    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_' + str(num_particles) + '_iters_' + str(max_iters))
    # path = './saved_items/50Nodes/'
    # graph_filename = path + 'graph_N50_cutoff6'
    # path_filename = graph_filename + 'paths'
    #
    #
    # sigma = 0.005
    # lr = 0.05
    # Niid = 10
    # max_iters = 500
    # num_particles = 10
    # decreasing_lr = True
    # eps = 1
    # seed = 0
    #
    # path = path + 'Results/' + graph_filename[-7:]
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #
    # args = {'graph_filename': graph_filename, 'path_filename': path_filename, 'sigma': sigma,
    #  'lr': lr, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # args_imd = {'graph_filename': graph_filename, 'path_filename': path_filename, 'sigma': sigma,
    #  'lr': lr, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    #
    #
    # losses_gd = run_GD(**args)
    # save_obj((losses_gd, args), path + '/GD_Niid_' + str(Niid) + '_iters_' + str(max_iters))
    # losses_md = run_MD(**args)
    # save_obj((losses_md, args), path + '/MD_Niid_' + str(Niid) + '_iters_' + str(max_iters))
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_' + str(num_particles) + '_iters_' + str(max_iters))
