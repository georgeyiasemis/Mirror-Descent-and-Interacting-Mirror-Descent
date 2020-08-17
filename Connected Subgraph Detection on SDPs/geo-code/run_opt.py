from geo_exps import *
import sys
import os
sys.path.insert(0, '../')
from helper_functions.save_load import save_obj
from startOpt.computation import computeAUC

n_nodes = 1000
radius = 0.5
seed = 1
# Ks = [5, 10, 15, 20, 25, 30, 40]
K =  15
thinRatio = 1
gammas = np.array([np.sqrt(2 / 1000)])
K = 20
# max_iter = 50
max_iters = [60, 65, 70]
nSamp = 40
num_particles = 2
Niid = 2
l_0 = 100
opt = 'IMD'
if opt == 'IMD':
    path = "../saved_data/geo/{}/N_{}nSamp_{}_Np_{}_K_{}".format(opt, n_nodes,nSamp, num_particles, K)
else:
    path = "../saved_data/geo/{}/N_{}nSamp_{}_Niid_{}_K_{}".format(opt, n_nodes,nSamp, Niid, K)

if not os.path.exists(path):
    os.mkdir(path)
aucs = []
for max_iter in max_iters:
    np.random.seed(0)
    l_1 = 100
    print('l1/l0={}'.format(l_1/l_0))
    if opt == 'MD':
        args = {'n_nodes': n_nodes, 'radius': radius, 'l_0': l_0, 'l_1': l_1, 'K': K,
            'thinRatio': thinRatio, 'gammas': gammas, 'max_iter': max_iter, 'nSamp': nSamp,
            'Niid': Niid, 'seed': seed}
        scores_noise = geo_exps_MD(**args)
    else:
        args = {'n_nodes': n_nodes, 'radius': radius, 'l_0': l_0, 'l_1': l_1, 'K': K,
            'thinRatio': thinRatio, 'gammas': gammas, 'max_iter': max_iter, 'nSamp': nSamp,
            'num_particles': num_particles, 'seed': seed}
        scores_noise = geo_exps_IMD(**args)

    save_obj((scores_noise, args), path + '/scores_noise_iters{}'.format(max_iter))

    l_1 = 110
    print('l1/l0={}'.format(l_1/l_0))
    if opt == 'MD':
        args = {'n_nodes': n_nodes, 'radius': radius, 'l_0': l_0, 'l_1': l_1, 'K': K,
            'thinRatio': thinRatio, 'gammas': gammas, 'max_iter': max_iter, 'nSamp': nSamp,
            'Niid': Niid, 'seed': seed}
        scores_signal = geo_exps_MD(**args)
    else:
        args = {'n_nodes': n_nodes, 'radius': radius, 'l_0': l_0, 'l_1': l_1, 'K': K,
            'thinRatio': thinRatio, 'gammas': gammas, 'max_iter': max_iter, 'nSamp': nSamp,
            'num_particles': num_particles, 'seed': seed}
        scores_signal = geo_exps_IMD(**args)


    save_obj((scores_signal, args), path + '/scores_signal{}_iters{}'.format(l_1/l_0, max_iter))

    auc = computeAUC(scores_noise, scores_signal, gammas).item()
    print('Iters: {}. AUC: {}'.format(max_iter, auc))
    aucs.append((max_iter, auc))
save_obj(aucs, path + '/aucs_iters_{}'.format(max_iter))
