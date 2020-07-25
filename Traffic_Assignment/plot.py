import matplotlib.pyplot as plt
from graph import Graph
import numpy as np
from helper_functions.save_load import load_obj

plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

path = "./saved_items/50Nodes/Results/"
path_gd = path + "cutoff5/GD_Niid_2_iters_500"
path_md = path + "cutoff5/MD_Niid_2_iters_500"
path_imd = path + "cutoff5/IMD_Np_10_iters_500"

gd_losses = load_obj(path_gd)[0]
md_losses = load_obj(path_md)[0]
imd_losses = load_obj(path_imd)[0]
sigma = load_obj(path_gd)[1]['sigma']
Niid = load_obj(path_gd)[1]['Niid']
num_particles = load_obj(path_imd)[1]['num_particles']
lr = load_obj(path_imd)[1]['lr']
decreasing_lr = load_obj(path_imd)[1]['decreasing_lr']
print(sigma, Niid, num_particles, lr, decreasing_lr)

plt.clf()
plt.semilogy(np.array(gd_losses),'r', label='SGD. $ Niid = $ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
plt.semilogy(np.array(md_losses),'b', label='SMD. $Niid =$ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
plt.semilogy(np.array(imd_losses),'k', label='SIMD. $N_p =$ {:3}, $\sigma =$ {:4}'.format(num_particles, sigma))
plt.xlabel('Iterations', fontsize = 20)
plt.ylabel('Value', fontsize = 20)
plt.title('Loss function for $|\mathcal{V}| = 50$, $n_{\mathcal{R}} = 70$', fontsize = 20)
plt.legend()
plt.tight_layout()
plt.savefig(path + "GD_IMD_sigma_{}.png".format(sigma), dpi=1000)
plt.show()
#
plt.figure()

plt.clf()
# title = 'Histogram after 100 iterations for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)

plt.hist(np.array(imd_losses)[50:], alpha=0.5, color='r',linewidth=0.5, label='SIMD')
plt.hist(np.array(md_losses)[30:], alpha=0.7, color='b',linewidth=0.5, label='SMD')
plt.hist(np.array(gd_losses)[50:], alpha=0.5, color='b',linewidth=0.5, label='SGD')
plt.savefig(path + "Hist_GD_MD_IMD_sigma_{}.png".format(sigma), dpi=1000)
plt.legend()
plt.show()
