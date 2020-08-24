import matplotlib.pyplot as plt
from graph import Graph
import numpy as np
from helper_functions.save_load import load_obj

plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

path = "./saved_items/100Nodes/Results/"
path_gd = path + "cutoff7/GD_Niid_2_iters_500_dec_lr_0.01"
path_md = path + "cutoff7/MD_Niid_2_iters_500_dec_lr_0.5"
path_imd = path + "cutoff7/IMD_Np_2_iters_500_dec_lr_0.5"

gd_losses = load_obj(path_gd)[0]
md_losses = load_obj(path_md)[0]
imd_losses = load_obj(path_imd)[0]


sigma = load_obj(path_gd)[1]['sigma']
Niid = load_obj(path_gd)[1]['Niid']
lr = load_obj(path_gd)[1]['lr']
decreasing_lr = load_obj(path_gd)[1]['decreasing_lr']
print('GD')
print(sigma, Niid, lr, decreasing_lr)

sigma = load_obj(path_md)[1]['sigma']
Niid_md = load_obj(path_md)[1]['Niid']
lr = load_obj(path_md)[1]['lr']
decreasing_lr = load_obj(path_md)[1]['decreasing_lr']
print('MD')
print(sigma, Niid_md, lr, decreasing_lr)

num_particles = load_obj(path_imd)[1]['num_particles']
sigma = load_obj(path_imd)[1]['sigma']
lr = load_obj(path_imd)[1]['lr']
decreasing_lr = load_obj(path_gd)[1]['decreasing_lr']
print('IMD')
print(sigma, num_particles, lr, decreasing_lr)

plt.clf()
plt.semilogy(np.array(gd_losses),'r', label=r'SGD, $\sigma$ = {}. $ Niid = $ {:3}'.format(sigma,Niid))
plt.semilogy(np.array(md_losses),'b', label=r'SMD, $\sigma$ = {}. $ Niid =$ {:3}'.format(sigma, Niid_md))
plt.semilogy(np.array(imd_losses),'k', label=r'SIMD, $\sigma$ = {}. $ N_p =$ {:3}'.format(sigma, num_particles))
plt.xlabel('Iterations', fontsize = 20)
plt.ylabel('Value', fontsize = 20)
# plt.ylim(min(imd_losses) -0.05, 7.5 )
plt.title('Loss function for $|\mathcal{V}| = 100$, $n_{\mathcal{R}} = 1647$', fontsize = 20)
plt.legend()
plt.tight_layout()
plt.savefig(path + "SGD_SMD_SIMD_dec.png")
plt.show()
# #
plt.figure()

plt.clf()
title = 'Histogram for $|\mathcal{V}| = 100$, $n_{\mathcal{R}} = 1647$'

plt.hist(np.array(gd_losses), color='r', alpha=0.7, label=r'SGD, $\sigma$ = {}. $ Niid = $ {:3}'.format(sigma,Niid))
plt.hist(np.array(md_losses), color='b', alpha=0.7, label=r'SMD, $\sigma$ = {}. $ Niid =$ {:3}'.format(sigma, Niid_md))
plt.hist(np.array(imd_losses), color='k', alpha=0.7, label=r'SIMD, $\sigma$ = {}. $ N_p =$ {:3}'.format(sigma, num_particles))
plt.ylabel('Frequency', fontsize = 20)
plt.legend()
plt.title(title, fontsize = 20)
plt.savefig(path + "Hist_SGD_SMD_SIMD_dec.png")

print(np.array(gd_losses).var(),
        np.array(md_losses).var(),
        np.array(imd_losses).var())
