# import numpy as np
# import matplotlib.pyplot as plt
#
# n_samples = 200
# np.random.seed(0)
# t = np.linspace(0, 10, n_samples)
# y = []
# angles = np.linspace(0, 2 * np.pi, 6)
# XX = []
# for (i,c) in enumerate(angles):
#
#     X1 = t * np.cos(t+c)
#     X2 = t * np.sin(t+c)
#
#
#     X = np.concatenate((X1.reshape(1,-1), X2.reshape(1,-1)), axis=0)
#     # X += .1 * np.random.randn(2, n_samples)
#     X = X.T
#     XX.append(X)
#     y.append(np.ones(X.shape[0]) * i)
#
#     plt.scatter(X[:,0], X[:,1])
#
# XX = np.array(XX).reshape(-1,2)
# print(XX.shape)
# y = np.array(y).reshape(-1)
# print(y)
# plt.savefig('figgg.png')

import os
import numpy as np
import torch
from run_opt import *
import matplotlib.pyplot as plt
if __name__ == "__main__":

    eps = 1
    max_iters = 50
    N = 1800
    D = 2
    K = 4
    lambd = 1
    var = 0.01
    cluster_std = 0.005
    cluster_random_state = 0
    sigma = 0
    seed = 11
    lr_gd = 0.00001
    lr_md = 0.00001
    lr_imd = 0.00001
    decreasing_lr = True
    Niid = 2
    num_particles = 1
    mode = 'spirals'
    args_gd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_gd, 'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}
    args_md = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_md,'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}
    args_imd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_imd, 'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}


    path = './N_{}_D_{}_K_{}_lambda_{}_var_{}_clusterstd_{}_{}/'.format(N, D, K, lambd, var, cluster_std, mode)

    if not os.path.exists(path):
        os.mkdir(path)

    path_imd = path + 'IMD_Np_{}_lr_{}_iters_{}'.format(num_particles, lr_md, max_iters)
    losses_imd, kmodes_imd = run_IMD(**args_imd)
    y = kmodes_imd.Z.argmax(axis=1)
    plt.figure()
    kmodes_imd.plot_2d_clusters('1111',y=y, C=kmodes_imd.C)
    torch.save(kmodes_imd, path_imd + '_model.pt')
    save_obj((losses_imd, args_imd), path_imd)
