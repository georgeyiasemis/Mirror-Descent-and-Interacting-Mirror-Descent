import matplotlib.pyplot as plt
from graph import Graph
import numpy as np
from helper_functions.save_load import load_obj

plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

path = "./saved_items/50Nodes/Results/"
path_gd = path + "cutoff5/GD_Niid_5_iters_500"
path_md = path + "cutoff5/MD_Niid_5_iters_500"
path_imd = path + "cutoff5/IMD_Np_5_iters_500"

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
plt.semilogy(np.array(gd_losses),'r', label='GD. $ Niid = $ {:3}'.format(Niid))
plt.semilogy(np.array(md_losses),'b', label='MD. $Niid =$ {:3}'.format(Niid))
plt.semilogy(np.array(imd_losses),'k', label='IMD. $N_p =$ {:3}'.format(num_particles))
plt.xlabel('Iterations', fontsize = 20)
plt.ylabel('Value', fontsize = 20)
plt.ylim(min(imd_losses) -0.05, 7.5 )
plt.title('Loss function for $|\mathcal{V}| = 50$, $n_{\mathcal{R}} = 70$', fontsize = 20)
plt.legend()
plt.tight_layout()
plt.savefig(path + "GD_MD_IMD_sigm.png", dpi=1000)
plt.show()
#
plt.figure()

plt.clf()
title = 'Histogram after 50 iterations for $|\mathcal{V}| = 50$, $n_{\mathcal{R}} = 70$'

plt.hist(np.array(gd_losses)[20:], color='r', alpha=0.7, label='GD. $ Niid = $ {:3}'.format(Niid))
plt.hist(np.array(md_losses)[20:], color='b', alpha=0.7, label='MD. $Niid =$ {:3}'.format(Niid))
plt.hist(np.array(imd_losses)[20:], color='k', alpha=0.7, label='IMD. $N_p =$ {:3}'.format(num_particles))
plt.ylabel('Frequency', fontsize = 20)
plt.legend()
plt.title(title, fontsize = 20)
plt.savefig(path + "Hist_GD_MD_IMD.png", dpi=1000)
