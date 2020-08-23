import os
import numpy as np
import torch
from helper_functions.projections import projection_simplex
from helper_functions.save_load import save_obj
from linear import LinearSystem
from tqdm import trange

def run_GD(m: int, d: int, condition: float, sigma: float, lr: float,
            max_iters=1000, Niid=10, decreasing_lr=False, seed=None):
    """Runs Gradient Descent Optimisation for Equation (5.3) of the project:

        \min_{x \in Simplex} f(x) = || W @ x - b ||_2^2,
                W \in \R^{m * d}, b_m \in \R^d.


    Parameters
    ----------
    m : int
        Total number of observations.
    d : int
        Dimension of the problem.
    condition : float
        Condition number of W.
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

    Returns
    -------
    iters: list
        List  of length max_iters containing the value of the loss function at
        each iteration.

    """

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (m > 0) and (d > 0), "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(m, d, condition, sigma)

    print()
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
            # x_0 = 1_d / d
            x_t = torch.ones((d, 1)).float() / d
            system.eval(x_t)

            iters[0] += system.loss / Niid

            for t in range(1, max_iters):
                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()

                # x_{t+1} = x_t - h_t * e * grad_f(x_t)
                x_t = x_t.reshape(-1, 1) - lr * lr_const * system.gradient.reshape(-1, 1)

                # x_{t+1} = Π_X(x_{t+1})
                x_t = projection_simplex(x_t)

                system.eval(x_t)

                iters[t] += system.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])

    return iters

def run_MD(m: int, d: int, condition: float, sigma: float, lr: float,
            max_iters=1000, Niid=10, decreasing_lr=False, seed=None):

    """Runs Mirror Descent Optimisation for Equation (5.3) of the project:

        \min_{x \in Simplex} f(x) = || W @ x - b ||_2^2,
                W \in \R^{m * d}, b_m \in \R^d.

    Parameters
    ----------
    m : int
        Total number of observations.
    d : int
        Dimension of the problem.
    condition : float
        Condition number of W.
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

    Returns
    -------
    iters: list
        List  of length max_iters containing the value of the loss function at
        each iteration.

    """

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (m > 0) and (d > 0), "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(m, d, condition, sigma)

    print()
    print('MD')
    print('--')
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

            system.eval(x_t)

            iters[0] += system.loss / Niid
            for t in range(1, max_iters):

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t).item()
                x_t = x_t.reshape(-1, 1) * torch.exp(- lr * lr_const * system.gradient).reshape(-1, 1)

                x_t = x_t / torch.sum(x_t)

                system.eval(x_t)

                iters[t] += system.loss / Niid

            tqdm.set_postfix(Loss=iters[-1])


    return iters

def run_IMD(m: int, d: int, condition: float, sigma: float, lr: float,
            max_iters=1000, num_particles=5, decreasing_lr=False, seed=None):

    """Runs Interacting Mirror Descent Optimisation for Equation (5.3) of the project:

        \min_{x \in Simplex} f(x) = || W @ x - b ||_2^2,
                W \in \R^{m * d}, b_m \in \R^d.

    Parameters
    ----------
    m : int
        Total number of observations.
    d : int
        Dimension of the problem.
    condition : float
        Condition number of W.
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

    Returns
    -------
    iters: list
        List  of length max_iters containing the value of the loss function at
        each iteration.

    """

    assert sigma >= 0.0, "Sigma should be non negative."
    assert (m > 0) and (d > 0), "Dimensions should be positive integers."
    assert condition == None or condition > 0

    iters = [0 for i in range(max_iters)]
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    lr_const = 1
    system = LinearSystem(m, d, condition, sigma)

    print()
    print()
    print('InterMD')
    print('-------')
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
            # A = particles

            for p in range(num_particles):
                x_p = particles[:, p]
                system.eval(x_p)
                obj_avg += system.loss / num_particles

                obj_min = system.loss if obj_min > system.loss else obj_min

                x_p = torch.prod(torch.pow(particles, interaction_mat[p]), axis=1)

                if decreasing_lr:
                    lr_const = 1 / np.sqrt(t + 1).item()

                x_p = x_p * torch.exp(- lr * lr_const * system.gradient).reshape(-1)

                x_p /= torch.sum(x_p)

                particles[:, p] = x_p

            iters[t] = obj_avg

            tqdm.set_postfix(Loss=iters[t])

    return iters

if __name__ == "__main__":
    sigma = 0.5
    max_iters = 1000
    m = 1000
    d = 1000
    condition = 10
    seed = 0
    lr_gd = 0.005
    lr_md = 0.1
    decreasing_lr = True
    Niid = 10
    num_particles = 1

    args_gd = {'m': m, 'd': d, 'condition': condition, 'sigma': sigma, 'lr': lr_gd,
        'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_md = {'m': m, 'd': d, 'condition': condition, 'sigma': sigma, 'lr': lr_md,
        'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed}
    args_imd = {'m': m, 'd': d, 'condition': condition, 'sigma': sigma, 'lr': lr_md,
         'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed}

    path = './saved_items/cond_' + str(condition) + 'd_'+ str(d)
    if not os.path.exists(path):
        os.mkdir(path)

    # losses_gd = run_GD(**args_gd)
    # save_obj((losses_gd, args_gd), path + '/GD_Niid_{}_iters_{}_s_{}'.format(Niid, max_iters, sigma))
    # losses_md = run_MD(**args_md)
    # save_obj((losses_md, args_md), path + '/MD_Niid_{}_iters_{}_s_{}'.format(Niid, max_iters, sigma))
    losses_imd = run_IMD(**args_imd)
    # save_obj((losses_imd, args_imd), path + '/IMD_Np_{}_iters_{}_s_{}'.format(num_particles, max_iters, sigma))
