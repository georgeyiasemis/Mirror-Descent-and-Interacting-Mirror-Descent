import numpy as np
import sys
sys.path.insert(0, '../')
import os
from helper_functions.genCounty import genCounty
from helper_functions.save_load import save_obj
from startOpt.startOpt import runOpt_imd, runOpt_md
from startOpt.computation import computeAUC
from tqdm import trange


def county_auc_MD(signal, gammas_n, max_iter=100, nSamp=50, Niid=10):
    gammas = np.logspace(-2,np.log10(5), gammas_n)


    A, s, yy = genCounty(signal, nSamp)
    scores_noise = np.zeros((Niid, nSamp, gammas_n))

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

def county_auc_IMD(signal, gammas_n, max_iter=100, nSamp=50, num_particles=5):
    gammas = np.logspace(-2,np.log10(5), gammas_n)

    A, s, yy = genCounty(signal, nSamp)
    scores_noise = np.zeros((nSamp, gammas_n))

    with trange(nSamp, ncols=100) as tqdm:
        for ns in tqdm:
            ys = yy[:,ns]
            c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])

            C = c.reshape(-1,1) @ c.reshape(1,-1)

            for gind in range(gammas_n):

                tqdm.set_description('IMD Np={} || gamma = {:2f}'.format(num_particles, gammas[gind]))

                M = runOpt_imd(A, C, gammas[gind], s, max_iter, num_particles=num_particles)

                scores_noise[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
                tqdm.set_postfix(Loss='{:8f}'.format(np.trace(C.T @ M)))

    return scores_noise


if __name__ == '__main__':
    signals = [1, 1.1, 1.3, 1.5, 1.7]
    gammas_n = 10
    gammas = np.logspace(-2,np.log10(5), gammas_n)
    max_iter = 50
    nSamp = 50
    Niid = 5
    num_particles = 2


    path = "../saved_data/county/IMD/nSamp_{}_Np_{}_Iters_{}_Ng_{}".format(nSamp, num_particles, max_iter, gammas_n)

    np.random.seed(1)
    if not os.path.exists(path):
        os.mkdir(path)

    aucs = []
    for signal in signals:
        print('Signal: {}\n'.format(signal))

        if signal == 1:
            scores_noise = county_auc_IMD(signal, gammas_n, max_iter, nSamp,  num_particles)
            save_obj(scores_noise, path + '/scores_noise')

        else:
            scores_signal = county_auc_IMD(signal, gammas_n, max_iter, nSamp,  num_particles)
            save_obj(scores_signal, path + '/scores_signal' + str(signal))
            auc = computeAUC(scores_noise, scores_signal, gammas)
            print('Max AUC:', auc.max())

            aucs.append((signal, auc))
    save_obj(aucs, path + '/aucs_pickle')

    path = "../saved_data/county/MD/nSamp_{}_Niid_{}_Iters_{}_Ng_{}".format(nSamp, Niid, max_iter, gammas_n)

    np.random.seed(1)
    if not os.path.exists(path):
        os.mkdir(path)

    aucs = []
    for signal in signals:
        print('Signal: {}\n'.format(signal))

        if signal == 1:
            scores_noise = county_auc_MD(signal, gammas_n, max_iter, nSamp,  Niid)
            save_obj(scores_noise, path + '/scores_noise')

        else:
            scores_signal = county_auc_MD(signal, gammas_n, max_iter, nSamp,  Niid)
            save_obj(scores_signal, path + '/scores_signal' + str(signal))
            auc = computeAUC(scores_noise, scores_signal, gammas)
            print('Max AUC:', auc.max())

            aucs.append((signal, auc))
    save_obj(aucs, path + '/aucs_pickle')
