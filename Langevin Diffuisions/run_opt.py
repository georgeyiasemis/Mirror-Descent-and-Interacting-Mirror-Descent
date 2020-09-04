import torch
import numpy as np
from tqdm import tqdm,  trange


def run_NLA(d: int, epsilon: float, max_iters: int, Niid: int):
    """
    Solves NLA (Newton-Langevin Algorithm) from Equation (5.14) of the project

        \ nabla V(X_{k+1}) = (1-epsilon)* \ nabla V(X_{k})
                + sqrt(2*epsilon) * [\ nabla^2 V(X_{k+1})]^(1/2) * \ xi_k

    where V(x) = 0.5 * (x^T * \ Sigma^(-1) * x), \ xi_k ~ N(0, I_d)

    and \ Sigma = diag{1,...,d}.



    Parameters
    ----------
    d : int
        Dimension of disired sampling space.
    epsilon : float
        Step size.
    max_iters : int
        Maximum Iterations to run optimisation.
    Niid : int
        Number of iid runs.

    Returns
    -------
    iters : dict
        Dcitionary containing running estimates for the mean  (log10(||\ bar{X_t}||_2^2))
        and scatter matrix  ||\ hat{\Sigma}-\ Sigma||_2^2 / ||\ Sigma||_2^2.

    """


    iters = {'mean_error': [0 for i in range(max_iters)], 'scatter_error': [0 for i in range(max_iters)]}
    Sigma = torch.diag(torch.arange(1,d+1)).float()

    with trange(Niid, ncols=150) as tqdm:

        x_t = torch.zeros(d).float()

        X = torch.zeros((max_iters,d)).float() # Store samples

        Sigmas = torch.zeros((max_iters, d, d)).float() # Store scatter matrices

        for i in tqdm:

            tqdm.set_description('MLD')

            for t in range(0, max_iters):
                x_t = (1-epsilon) * x_t + np.sqrt(2 * epsilon) * torch.pow(Sigma, 1/2) @ torch.randn(d).reshape(-1)
                X[t] = x_t
                Sigmas[t] = torch.tensor(np.cov(X[:t+1].mean(0).numpy()))

                mean = torch.log10(torch.norm(X[:t+1].mean(0))**2)
                scatter = torch.log10(torch.norm(Sigmas[:t+1].mean(0)-Sigma)**2/torch.norm(Sigma)**2)

                iters['mean_error'][t] += mean / Niid
                iters['scatter_error'][t] += scatter / Niid

            tqdm.set_postfix(Loss=iters['mean_error'][-1], Loss_scatter=iters['scatter_error'][-1])

    return iters

def run_INLA(d: int, epsilon: float, max_iters: int, num_particles: int):
    """
    Solves INLA (Interacting Newton-Langevin Algorithm) from Equation (5.17) of the project

        \ nabla V(X_{k+1}^i) = (1 - 2*epsilon)* \ nabla V(X_{k}^i)
                + epsilon \ sum_{j=1}^Np A_{ij}  \ nabla V(X_{k}^j)
                + sqrt(2*epsilon) * [\ nabla^2 V(X_{k+1}^i)]^(1/2) * \ xi_k^i,
        i = 1,..,Np

    where V(x) = 0.5 * (x^T * \ Sigma^(-1) * x), \ xi_k^i ~ N(0, I_d)

    and \ Sigma = diag{1,...,d}.



    Parameters
    ----------
    d : int
        Dimension of disired sampling space.
    epsilon : float
        Step size.
    max_iters : int
        Maximum Iterations to run optimisation.
    num_particles : int
        Number of interacting particles.

    Returns
    -------
    iters : dict
        Dcitionary containing running estimates for the mean  (log10(||\ bar{X_t}||_2^2))
        and scatter matrix  ||\ hat{\Sigma}-\ Sigma||_2^2 / ||\ Sigma||_2^2.

    """


    iters = {'mean_error': [0 for i in range(max_iters)], 'scatter_error': [0 for i in range(max_iters)]}
    Sigma = torch.diag(torch.arange(1,d+1)).float()

    particles = torch.rand(d, num_particles).float()
    # particles = particles / particles.sum(0)

    interaction_mat = torch.ones((num_particles,
                                num_particles)).float() / num_particles


    X = torch.zeros((max_iters,d, num_particles)).float()
    Sigmas = torch.zeros((max_iters, d, d, num_particles)).float()
    with trange(max_iters, ncols=150) as tqdm:
        for t in tqdm:
            tqdm.set_description('IMLD')
            obj_avg = 0.0
            obj_avg_scatter = 0.0


            for p in range(num_particles):
                x_p = particles[:, p]
                x_p = (1 - 2*epsilon) * x_p + np.sqrt(2 * epsilon) * torch.pow(Sigma, 1/2) @ torch.randn(d)

                x_p += epsilon * interaction_mat[p,:] @ particles.T

                X[t,:,p] = x_p

                loss = torch.log10(torch.norm(X[:t+1,:,p].mean(0))**2)

                Sigmas[t,:,:,p] = torch.tensor(np.cov(X[:t+1,:,p].mean(0).numpy()))

                scatter = Sigmas[:t+1,:,:,p].mean(0)
                loss_scatter = torch.log10(torch.norm(scatter - Sigma)**2 / torch.norm(Sigma)**2)

                obj_avg += loss / num_particles

                obj_avg_scatter += loss_scatter / num_particles

                particles[:, p] = x_p

            iters['mean_error'][t] = obj_avg
            iters['scatter_error'][t] = obj_avg_scatter
            tqdm.set_postfix(Loss=iters['mean_error'][t], Loss_scatter=iters['scatter_error'][t])


    return iters

if __name__ == '__main__':
    d = 100
    epsilon = 0.7
    max_iters = 2000
    Niid = 10
    num_particles = 10

    args_NLA = {'d': d, 'epsilon': epsilon, 'max_iters': max_iters, 'Niid': Niid}

    args_INLA = {'d': d, 'epsilon': epsilon, 'max_iters': max_iters, 'num_particles': num_particles}

    iters_NLA = run_NLA(**args_NLA)

    iters_INLA = run_INLA(**args_INLA)

    torch.save([iters_NLA, args_NLA], './saved_items/d_100/iters_NLA.pkl')
    torch.save([iters_INLA, args_INLA], './saved_items/d_100/iters_INLA.pkl')
