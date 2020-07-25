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
path_simd1 = "./saved_items/cond_200d_1000/IMD_Np_1_iters_1000_s_0.5"
path_simd10 = "./saved_items/cond_200d_1000/IMD_Np_10_iters_1000_s_0.5"
path_simd50 = "./saved_items/cond_200d_1000/IMD_Np_50_iters_1000_s_0.5"
path_simd100 = "./saved_items/cond_200d_1000/IMD_Np_100_iters_1000_s_0.5"

#
# gd_losses = load_obj(path_gd)[0]
# sgd_losses = load_obj(path_sgd)[0]
# md_losses = load_obj(path_md)[0]
# imd_losses = load_obj(path_imd)[0]
simd_losses1 = load_obj(path_simd1)[0]
simd_losses10 = load_obj(path_simd10)[0]
simd_losses50 = load_obj(path_simd50)[0]
simd_losses100 = load_obj(path_simd100)[0]
print(np.array(simd_losses1)[200:].var())
print(np.array(simd_losses10)[200:].var())
print(np.array(simd_losses50)[200:].var())
print(np.array(simd_losses100)[200:].var())
sigma = load_obj(path_simd100)[1]['sigma']
# Niid = load_obj(path_smd)[1]['Niid']
num_particles = load_obj(path_simd100)[1]['num_particles']
lr = load_obj(path_simd10)[1]['num_particles']
print(lr)
# decreasing_lr = load_obj(path_imd)[1]['decreasing_lr']
condition = load_obj(path_simd100)[1]['condition']
d = load_obj(path_simd100)[1]['d']
# print(sigma, Niid, num_particles, lr, decreasing_lr)
path = './figures/cond_' + str(condition) + 'd_' + str(d)
if not os.path.exists(path):
    os.mkdir(path)
#
# fig, axs = plt.subplots(1, 2,True)
# fig.set_size_inches(11,4)

# axs[0,0].semilogy(np.array(sgd_losses),'r', label='SGD. $ Niid = $ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
# axs[0,0].semilogy(np.array(gd_losses),'k',  label='GD. $Niid $ = {:3}'.format(Niid))

#
# for (ax) in axs.flatten():
#
#     ax.set_xlabel('Iterations', fontsize = 10)
#     ax.set_ylabel('Function Value', fontsize = 10)
#     ax.legend()
# fig.suptitle(title, fontsize = 20)
plt.figure(figsize=(8,6))
plt.semilogy(np.array(simd_losses1),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(1, sigma))
plt.semilogy(np.array(simd_losses10),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(10, sigma))
plt.semilogy(np.array(simd_losses50),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(50, sigma))
plt.semilogy(np.array(simd_losses100),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(100, sigma))
plt.ylabel('Function Value')
plt.xlabel('Iterations')
title = 'Loss function for $n = {}$, $\kappa(W) = {}$'.format(d, condition)
plt.title(title, fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()
plt.ylim(min(simd_losses100)-2, 680)
plt.savefig(path + "/SIMD_Np_{}_sigma_{}.png".format(100,sigma), dpi=300)


plt.figure(figsize=(8,6))
plt.hist(np.array(simd_losses1)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(1, sigma))
plt.hist(np.array(simd_losses10)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(10, sigma))
plt.hist(np.array(simd_losses50)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(50, sigma))
plt.hist(np.array(simd_losses100)[200:],linewidth=0.5, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(100, sigma))
plt.legend()
title = 'Histogram after 200 iterations for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)
plt.title(title, fontsize=20)
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(path + "/Hist_SIDM_Np_{}_sigma_{}.png".format(100,sigma), dpi=300)
