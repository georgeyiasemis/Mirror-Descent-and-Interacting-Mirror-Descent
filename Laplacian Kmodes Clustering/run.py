import os
import numpy as np
import torch
from run_opt import *
import matplotlib.pyplot as plt
if __name__ == "__main__":

    eps = 1
    max_iters = 50
    N = 2000
    D = 3
    K = 10
    lambd = 100
    var = 10
    cluster_std = 0.7
    cluster_random_state = 0
    sigma = 0
    seed = 1
    lr_gd = 0.00001
    lr_md = 0.00001
    lr_imd = 0.001
    decreasing_lr = True
    Niid = 2
    num_particles = 2
    mode = 'blobs'
    args_gd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_gd, 'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}
    args_md = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_md,'eps': eps, 'max_iters': max_iters, 'Niid': Niid, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}
    args_imd = {'N': N, 'D': D, 'K': K, 'lambd': lambd, 'var': var, 'cluster_std': cluster_std, 'cluster_random_state': cluster_random_state,
        'sigma': sigma, 'lr': lr_imd, 'eps': eps, 'max_iters': max_iters, 'num_particles': num_particles, 'decreasing_lr': decreasing_lr, 'seed': seed, 'mode': mode}


    path = './saved_items/N_{}_D_{}_K_{}_lambda_{}_var_{}_clusterstd_{}_{}/'.format(N, D, K, lambd, var, cluster_std, mode)

    if not os.path.exists(path):
        os.mkdir(path)

    path_imd = path + 'IMD_Np_{}_lr_{}_iters_{}'.format(num_particles, lr_md, max_iters)
    losses_imd, kmodes_imd = run_IMD(**args_imd)
    # y = kmodes_imd.Z.argmax(axis=1)
    # plt.figure()
    # kmodes_imd.plot_3d_clusters('1111',y=y, C=kmodes_imd.C)
    torch.save(kmodes_imd, path_imd + '_model.pt')
    save_obj((losses_imd, args_imd), path_imd)

    path_md = path + 'MD_Niid_{}_lr_{}_iters_{}'.format(Niid, lr_md, max_iters)
    losses_md, kmodes_md = run_MD(**args_md)
    torch.save(kmodes_md, path_md + '_model.pt')
    save_obj((losses_md, args_md), path_md)

    path_gd = path + 'GD_Niid_{}_lr_{}_iters_{}'.format(Niid, lr_gd, max_iters)
    losses_gd, kmodes_gd = run_GD(**args_gd)
    torch.save(kmodes_gd, path_gd + '_model.pt' )
    save_obj((losses_gd, args_gd), path_gd)
