import numpy as np
import sys
import scipy
import os
sys.path.insert(0, '../')
from genmeasurements import genMeasurements
from helper_functions.save_load import save_obj
from startOpt.startOpt import *
from tqdm import trange
from generate_graph import Geo_graph_3d
from helper_functions.save_load import save_obj


# path = "../geo-code/example_10k_3d.graph"
#
# A, pts = readGeoGraph(path)

# ------------------------------------------------------------------------------

def geo_exps_IMD(n_nodes, radius, l_0, l_1, K=40, thinRatio=1,
                gammas=10, max_iter=100, nSamp=50, num_particles=3, seed=0):


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
                M = startOpt_imd(A, C, gammas[gind], s, max_iter, num_particles)
                scores_noise[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
                tqdm.set_postfix(Loss='{:6f}'.format(np.trace(C.T @ M)))

    return scores_noise

def geo_exps_MD(n_nodes, radius, l_0, l_1, K=40, thinRatio=1,
                gammas=10, max_iter=100, nSamp=50, Niid=1, seed=0):


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

                    M = startOpt_md(A=A, C=C, gamma=gammas[gind], s=s, max_iter=max_iter)

                    scores[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
                    tqdm.set_postfix(Loss='{:8f}'.format(np.trace(C.T @ M)))

        scores_noise[niid] = scores

    return scores_noise.mean(0)

# if __name__ == '__main__':
    #
    # n_nodes = 1000
    # radius = 0.5
    # seed = 1
    #
    # gammas_n = 1
    # K = 20
    # max_iter = 10
    # nSamp = 40
    # num_particles = 3
    # Niid = 5
    # path = "../saved_data/geo/IMD/Nnodes_nSamp_{}_Np_{}_Iters_{}_Ng_{}".format(n_nodes,nSamp, num_particles, max_iter, gammas_n)
    #
    # if not os.path.exists(path):
    #     os.mkdir(path)
    #
    # l_0, l_1 = 100, 100
    # scores_noise = geo_exps_IMD(n_nodes, radius, l_0, l_1, K,1, gammas_n, max_iter, nSamp, num_particles, seed)
    # save_obj(scores_noise, path + '/scores_noise')
    # l_0, l_1 = 100, 150
    # scores_signal = geo_exps_IMD(n_nodes, radius, l_0, l_1, K,1, gammas_n, max_iter, nSamp, num_particles)
    # save_obj(scores_signal, path + '/scores_signal{}'.format(l_1/l_0))
