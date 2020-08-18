import numpy as np
import torch
from math import ceil
from projections import projection_simplex
import random as rand
from helper_functions.progress import progress
from helper_functions.save_load import save_obj
from linear import LinearSystem
import os
from tqdm import tqdm,  trange


def run_GD(M: int, d: int, batch_size: int, condition: float, sigma: float, lr: float,
            eps=1, max_iters=1000, Niid=10, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (M > 0) and (M >= batch_size)
    assert (d > 0) , "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(M, d, condition, sigma)

    print('GD')
    print('--')
    print('Batch Size: ', batch_size)
    print('Number of iid runs: ', Niid)
    print('sigma = ', sigma)

    with trange(Niid, ncols=150) as tqdm:
        for i in tqdm:
            if sigma > 0:
                tqdm.set_description('SGD')
            else:
                tqdm.set_description('GD')
            # x_0 = 1_d / d
            x_t = torch.ones((d, 1)).float() / d
            system.eval(x_t, batch_size)

            iters[0] += system.loss / Niid

            for t in range(1, max_iters):
                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()

                # x_{t+1} = x_t - h_t * e * grad_f(x_t)
                x_t = x_t.reshape(-1, 1) - lr * lr_const * eps* system.gradient.reshape(-1, 1)

                # x_{t+1} = Π_X(x_{t+1})
                x_t = projection_simplex(x_t)

                system.eval(x_t, batch_size)

                iters[t] += system.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])

    return iters

def run_MD(M: int, d: int, batch_size: int, condition: float, sigma: float, lr: float,
            eps=1, max_iters=1000, Niid=10, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (M > 0) and (M >= batch_size)
    assert (d > 0) , "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(M, d, condition, sigma)

    print('MD')
    print('--')
    print('Batch Size: ', batch_size)
    print('Number of iid runs: ', Niid)
    print('sigma = ', sigma)

    with trange(Niid, ncols=150) as tqdm:
        for i in tqdm:

            if sigma > 0:

                tqdm.set_description('SMD')
                x_t = torch.rand(d, 1).float()
                x_t = x_t / torch.sum(x_t)

            else:

                tqdm.set_description('MD')
                x_t = torch.ones((d, 1)).float() / d

            system.eval(x_t, batch_size)

            iters[0] += system.loss / Niid
            for t in range(1, max_iters):

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()
                x_t = x_t.reshape(-1, 1) * torch.exp(- lr * lr_const * eps * system.gradient).reshape(-1, 1)

                x_t = x_t / torch.sum(x_t)

                system.eval(x_t, batch_size)

                iters[t] += system.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])


    return iters

def run_IMD(M: int, d: int, batch_size: int, condition: float, sigma: float, lr: float,
            eps=1, max_iters=1000, num_particles=5, decreasing_lr=False, seed=None):

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (M > 0) and (M >= batch_size)
    assert (d > 0) , "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(M, d, condition, sigma)

    print()
    print('InterMD')
    print('-------')
    print('Batch Size: ', batch_size)
    print('Number of iteracting paricles: ', num_particles)
    print('sigma = ', sigma)

    assert num_particles >= 1

    particles = torch.rand(d, num_particles).float()
    particles = particles / particles.sum(0)

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

            for p in range(num_particles):
                x_p = particles[:, p]
                system.eval(x_p, batch_size)
                obj_avg += system.loss / num_particles

                obj_min = system.loss if obj_min > system.loss else obj_min

                x_p = torch.prod(torch.pow(particles, interaction_mat[p]), axis=1)

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t + 1).item()

                x_p = x_p * torch.exp(- lr * lr_const * eps * system.gradient).reshape(-1)

                x_p /= torch.sum(x_p)

                particles[:, p] = x_p

            iters[t] = obj_avg

            tqdm.set_postfix(Loss=iters[t])

    return iters



if __name__ == "__main__":
    sigma = 1
    eps = 1
    max_iters = 1000
    M = 100
    d = 500
    condition = 10
    seed = 10
    lr = 0.05
    decreasing_lr = True
    Niid = 10
    num_particles = 5
    batch_size = 5

    args = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
        'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
        'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}

    path = './saved_items/cond_' + str(condition)
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/d_'+ str(d)
    if not os.path.exists(path):
        os.mkdir(path)

    path = path + '/Compare_GD_MD_IMD'
    if not os.path.exists(path):
        os.mkdir(path)
    #
    # Compare All
    # #
    # losses_gd = run_GD(**args)
    # save_obj((losses_gd, args), path + '/GD_Niid_{}_iters_{}_M_{}_sigma_{}_batch_{}'.format(Niid, max_iters, M, sigma, batch_size))
    # losses_md = run_MD(**args)
    # save_obj((losses_md, args), path + '/MD_Niid_{}_iters_{}_M_{}_sigma_{}_batch_{}'.format(Niid, max_iters, M, sigma, batch_size))

    for num_particles in [1, 5, 10, 20, 50, 100]:
        args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
            'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}


        losses_imd = run_IMD(**args_imd)
        save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_sigma_{}_batch_{}'.format(num_particles, max_iters, M, sigma, batch_size))


    # Compare IMD batch vs full-batch

    # path = path + '/Compare_IMD'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #
    #
    # batch_size = 5
    # num_particles = 10
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    #
    # batch_size = 5
    # num_particles = 1
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    #
    # batch_size = 10
    # num_particles = 1
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    #
    # batch_size = 10
    # num_particles = 10
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    # batch_size = 15
    # num_particles = 1
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    #
    # batch_size = 15
    # num_particles = 10
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    #
    # batch_size = 30
    # num_particles = 10
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    # batch_size = 50
    # num_particles = 5
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
    #
    # batch_size = M
    # num_particles = 1
    # args_imd = {'M': M, 'd': d, 'batch_size': batch_size, 'condition': condition, 'sigma': sigma, 'lr': lr,
    #     'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}
    #
    # losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_M_{}_batch_{}'.format(num_particles, max_iters, M, batch_size))
