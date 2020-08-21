import sys
sys.path.insert(0, '../')
import numpy as np
from genmeasurements import genMeasurements
from startOpt.startOpt import *
from tqdm import trange
from generate_graph import Geo_graph_3d


def geo_exps_IMD(n_nodes, radius, l_0, l_1, K=40, thinRatio=1,
                gammas=10, max_iter=100, nSamp=50, num_particles=3, seed=0):

    """Solves the Connected Subgraph Detection problem and calculates AUC using
        Interacting Mirror Descent Optimisation for a random geometric graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes for the random graph.
    radius : float
        Distance threshold value.
    l_0 : float
        Base rate.
    l_1 : float
        Anomalous rate.
    K : int
        Anomaly size.
    thinRatio : float
        Ratio of max semi axis length to min semi axis length. Determines if graph is an ellipsoid or a sphere.
    gammas : int or np.array
        Conductance rates.
    max_iter : int
        Number of iterations.
    nSamp : int
        Number of samples.
    num_particles : int
        Number of interacting particles.
    seed : int
        Random seed.

    Returns
    -------
    scores_noise : np.array
        List of shape (nSamp, gammas_n) with AUC scores of optimisation.

    """


    graph = Geo_graph_3d(n_nodes=n_nodes, radius=radius, seed=seed)

    A, pts  = graph.Adj, graph.pos_array

    if type(gammas) == int:

        gammas = np.logspace(-3, np.log10(2), gammas)

    gammas_n = gammas.shape[0]

    yy, S = genMeasurements(pts, K, l_0, l_1, nSamp, thinRatio)

    s = S[0]
    scores_noise = np.zeros((nSamp, gammas_n), dtype='float32')

    with trange(nSamp, ncols=100) as tqdm:
        for ns in tqdm:

            ys = yy[:,ns]
            c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])

            C = c.reshape(-1,1) @ c.reshape(1,-1)

            for gind in range(gammas_n):
                tqdm.set_description('IMD || Np = {} gamma = {:2f}'.format(num_particles, gammas[gind]))
                M = runOpt_imd(A, C, gammas[gind], s, max_iter, num_particles)
                scores_noise[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
                tqdm.set_postfix(Loss='{:6f}'.format(np.trace(C.T @ M)))

    return scores_noise

def geo_exps_MD(n_nodes, radius, l_0, l_1, K=40, thinRatio=1,
                gammas=10, max_iter=100, nSamp=50, Niid=1, seed=0):

    """Solves the Connected Subgraph Detection problem and calculates AUC using
        Mirror Descent Optimisation for a random geometric graph.


    Parameters
    ----------
    n_nodes : int
        Number of nodes for the random graph.
    radius : float
        Distance threshold value.
    l_0 : float
        Base rate.
    l_1 : float
        Anomalous rate.
    K : int
        Anomaly size.
    thinRatio : float
        Ratio of max semi axis length to min semi axis length. Determines if graph is an ellipsoid or a sphere.
    gammas : int or np.array
        Conductance rates.
    max_iter : int
        Number of iterations.
    nSamp : int
        Number of samples.
    Niid : int
        Number of iid runs.
    seed : int
        Random seed.

    Returns
    -------
    scores_noise : np.array
        List of shape (nSamp, gammas_n) with AUC scores of optimisation.

    """



    graph = Geo_graph_3d(n_nodes=n_nodes, radius=radius, seed=seed)

    A, pts  = graph.Adj, graph.pos_array
    if type(gammas) == int:

        gammas = np.logspace(-3, np.log10(2), gammas)

    gammas_n = gammas.shape[0]

    yy, S = genMeasurements(pts, K, l_0, l_1, nSamp, thinRatio)

    s = S[0]
    scores_noise = np.zeros((Niid, nSamp, gammas_n), dtype='float32')

    for niid in range(Niid):
        print('No of iid run: {}'.format(niid+1))
        scores = np.zeros((nSamp, gammas_n))
        with trange(nSamp, ncols=100) as tqdm:
            for ns in tqdm:
                ys = yy[:,ns]
                c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])

                C = c.reshape(-1,1) @ c.reshape(1,-1)

                for gind in range(gammas_n):
                    tqdm.set_description('MD || Run = {} gamma = {:2f}'.format(niid+1, gammas[gind]))

                    M = runOpt_md(A=A, C=C, gamma=gammas[gind], s=s, max_iter=max_iter)

                    scores[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
                    tqdm.set_postfix(Loss='{:8f}'.format(np.trace(C.T @ M)))

        scores_noise[niid] = scores

    return scores_noise.mean(0)
