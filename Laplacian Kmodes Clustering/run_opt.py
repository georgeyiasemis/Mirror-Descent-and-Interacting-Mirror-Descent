import os
import numpy as np
import torch
from projections import projection_simplex
import random as rand
from helper_functions.save_load import save_obj
from kmodes import LaplacianKmodes
from tqdm import tqdm,  trange

def run_GD(N: int, D: int, K: int, lambd: float, var: float, cluster_std: float, cluster_random_state: int,
        sigma: float, lr: float, eps=1, max_iters=1000, Niid=10, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert N > 0, "Number of samples should be a positive integer."
    assert D > 0, "Dimension should be a positive integer."
    assert K > 0, "Number of clusters should be a positive integer."
    assert lambd > 0, "Lambda should be positive."

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    kmodes = LaplacianKmodes(N, D, K, lambd, var, cluster_std, cluster_random_state, sigma)

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

            Z_t = torch.ones((N, K)).float() / K
            kmodes.eval(Z_t)

            iters[0] += kmodes.loss / Niid

            for t in range(1, max_iters):
                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()

                # x_{t+1} = Z_t - h_t * e * grad_f(Z_t)
                Z_t = Z_t - lr * lr_const * eps* kmodes.gradient

                # x_{t+1} = Π_X(x_{t+1})
                for n in range(N):
                    Z_t[n] = projection_simplex(Z_t[n])

                kmodes.eval(Z_t)

                iters[t] += kmodes.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])

    return iters

def run_MD(N: int, D: int, K: int, lambd: float, var: float, cluster_std: float, cluster_random_state: int,
        sigma: float, lr: float, eps=1, max_iters=1000, Niid=10, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert N > 0, "Number of samples should be a positive integer."
    assert D > 0, "Dimension should be a positive integer."
    assert K > 0, "Number of clusters should be a positive integer."
    assert lambd > 0, "Lambda should be positive."

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    kmodes = LaplacianKmodes(N, D, K, lambd, var, cluster_std, cluster_random_state, sigma)

    print('MD')
    print('--')
    print('Number of iid runs: ', Niid)
    print('sigma = ', sigma)

    with trange(Niid, ncols=150) as tqdm:
        for i in tqdm:

            if sigma > 0:

                tqdm.set_description('SMD')
                Z_t = torch.rand(N, K).float()
                Z_t = Z_t / torch.sum(Z_t,1).reshape(-1,1)

            else:

                tqdm.set_description('MD')
                Z_t = torch.ones((N, K)).float() / K

            kmodes.eval(Z_t)

            iters[0] += kmodes.loss / Niid
            for t in range(1, max_iters):

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()
                Z_t = Z_t * torch.exp(- lr * lr_const * eps * kmodes.gradient)

                Z_t = Z_t / torch.sum(Z_t,1).reshape(-1,1)

                kmodes.eval(Z_t)

                iters[t] += kmodes.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])


    return iters

def run_IMD(N: int, D: int, K: int, lambd: float, var: float, cluster_std: float, cluster_random_state: int,
        sigma: float, lr: float, eps=1, max_iters=1000, num_particles=5, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert N > 0, "Number of samples should be a positive integer."
    assert D > 0, "Dimension should be a positive integer."
    assert K > 0, "Number of clusters should be a positive integer."
    assert lambd > 0, "Lambda should be positive."

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    kmodes = LaplacianKmodes(N, D, K, lambd, var, cluster_std, cluster_random_state, sigma)

    print()
    print('InterMD')
    print('-------')
    print('Number of iteracting paricles: ', num_particles)
    print('sigma = ', sigma)

    assert num_particles >= 1

    particles = torch.ones((N, K, num_particles)).float()
    particles = particles / particles.sum(1).reshape(particles.shape[0], 1, particles.shape[2])

    interaction_mat = torch.ones((num_particles,
                                num_particles)).float() / num_particles

    obj = float('Inf')

    with trange(max_iters, ncols=150) as tqdm:
        for t in tqdm:
            if sigma > 0:
                tqdm.set_description('SIMD')
            else:
                tqdm.set_description('IMD')

            obj_min = float('Inf')
            obj_avg = 0.0
            # A = particles

            for p in range(num_particles):
                Z_p = particles[:, :, p]

                kmodes.eval(Z_p)
                obj_avg += kmodes.loss / num_particles

                obj_min = kmodes.loss if obj_min > kmodes.loss else obj_min

                Z_p = torch.prod(torch.pow(particles, interaction_mat[p]), axis=2)

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t + 1).item()

                Z_p = Z_p * torch.exp(- lr * lr_const * eps * kmodes.gradient)

                Z_p /= torch.sum(Z_p,1).reshape(-1,1)

                particles[:, :, p] = Z_p

            iters[t] = obj_avg

            tqdm.set_postfix(Loss=iters[t])

    return iters

if __name__ == "__main__":
    eps = 1
    max_iters = 100
    N = 100
    D = 3
    K = 10
    lambd = 2
    var = 1
    cluster_std = 1.5
    cluster_random_state = 10
    sigma = 0
    seed = 1
    lr_gd = 0.0005
    lr_md = 0.1
    lr_imd = 0.1
    decreasing_lr = True
    Niid = 1
    num_particles = 2

    args_gd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_gd, 'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_md = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_md,'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_imd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_imd, 'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}

    path = './saved_items/Nsamp_{}/'.format(N)
    if not os.path.exists(path):
        os.mkdir(path)
    #
    # losses_gd = run_GD(**args_gd)
    # save_obj((losses_gd, args_gd), path + str(args_gd))
    losses_md = run_MD(**args_md)
    save_obj((losses_md, args_md), path + str(args_md))
    losses_imd = run_IMD(**args_imd)
    save_obj((losses_imd, args_imd), path + str(args_imd))
