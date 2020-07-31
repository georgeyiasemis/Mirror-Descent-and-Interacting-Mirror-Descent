import matplotlib.pyplot as plt
import numpy as np
from helper_functions.save_load import load_obj
import os


plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.2
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'



# path_gd = "./saved_items/cond_10d_1000/GD_Niid_10_iters_1000_s_0.0"
# path_md = "./saved_items/cond_10d_1000/MD_Niid_10_iters_1000_s_0.0"
# path_imd = "./saved_items/cond_10d_1000/IMD_Np_5_iters_1000_s_0.0"

# path_sgd = "./saved_items/cond_10d_1000/GD_Niid_10_iters_1000_s_0.5"
# path_smd = "./saved_items/cond_10d_1000/MD_Niid_10_iters_1000_s_0.5"
# path_sgd = "./saved_items/GD_Niid_2_iters_100_s_1"
path_sgd = "./saved_items/Nsamp_1000/{'N' 1000, 'D' 3, 'K' 10, 'lambd' 1, 'var' 1, 'cluster_std' 1.5, 'cluster_random_state' 0, 'sigma' 0, 'lr' 0.0005, 'eps' 1, 'max_iters' 2000, 'Niid' 5, 'decreasing_lr' True, 'seed' 0}"
# path_smd = "./saved_items/Nsamp_1000/{'N' 1000, 'D' 3, 'K' 10, 'lambd' 1, 'var' 1, 'cluster_std' 1.5, 'cluster_random_state' 0, 'sigma' 0, 'lr' 0.01, 'eps' 1, 'max_iters' 1000, 'Niid' 5, 'decreasing_lr' True, 'seed' 0}"
# path_smd = "./saved_items/MD_Niid_2_iters_100_s_1"
# path_imd = "./saved_items/Nsamp_1000/{'N' 1000, 'D' 3, 'K' 10, 'lambd' 1, 'var' 1, 'cluster_std' 1.5, 'cluster_random_state' 0, 'sigma' 0, 'lr' 0.005, 'eps' 1, 'max_iters' 1000, 'num_particles' 5, 'decreasing_lr' True, 'seed' 0}"

sgd_losses = load_obj(path_sgd)[0]
# smd_losses = load_obj(path_smd)[0]
# imd_losses = load_obj(path_imd)[0]

# Niid = load_obj(path_sgd)[1]['Niid']
args = load_obj(path_sgd)[1]


print(str(args))
path = './figures/'
if not os.path.exists(path):
    os.mkdir(path)

plt.figure(figsize=(8,6))
# plt.plot(np.array(smd_losses)[0:],linewidth=1, label='MD')
plt.plot(np.array(sgd_losses)[0:],linewidth=1, label='GD')
# plt.plot(np.array(imd_losses)[0:],linewidth=1, label='IMD')
# plt.semilogy(np.array(simd_losses50),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(50, sigma))
# plt.semilogy(np.array(simd_losses100),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(100, sigma))
plt.ylabel('Function Value')
plt.xlabel('Iterations')
title = 'Loss function for $N$ = {}, $K$ = {}, $D$ = {}'.format(args['N'], args['K'], args['D'])
plt.title(title, fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()
# plt.ylim(min(smd_losses)-2, 10)
plt.savefig(path + 'fig.png', dpi=300)

#
# plt.figure(figsize=(8,6))
# plt.hist(np.array(simd_losses1)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(1, sigma))
# plt.hist(np.array(simd_losses10)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(10, sigma))
# plt.hist(np.array(simd_losses50)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(50, sigma))
# plt.hist(np.array(simd_losses100)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(100, sigma))
# plt.legend()
# title = 'Histogram after 200 iterations for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)
# plt.title(title, fontsize=20)
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig(path + "/Hist_SIDM_Np_{}_sigma_{}.png".format(100,sigma), dpi=300)
