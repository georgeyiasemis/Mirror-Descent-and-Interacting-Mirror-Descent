import matplotlib.pyplot as plt
import numpy as np
from helper_functions.save_load import load_obj
import os


plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'



# path_gd = "./saved_items/cond_100/d_500/Compare_GD_MD_IMD/GD_Niid_10_iters_1000_M_100_sigma_2_batch_5"
path_md = "./saved_items/cond_10/d_500/Compare_GD_MD_IMD/MD_Niid_10_iters_1000_M_100_sigma_1_batch_5"

path_imd = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_1_iters_1000_M_100_sigma_1_batch_5"
path_imd1 = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_5_iters_1000_M_100_sigma_1_batch_5"
path_imd2 = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_10_iters_1000_M_100_sigma_1_batch_5"
path_imd3 = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_20_iters_1000_M_100_sigma_1_batch_5"
path_imd4 = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_50_iters_1000_M_100_sigma_1_batch_5"
path_imd5 = "./saved_items/cond_10/d_500/Compare_IMD/IMD_Np_100_iters_1000_M_100_sigma_1_batch_5"

# gd_losses = load_obj(path_gd)[0]
# md_losses = load_obj(path_md)[0]
# Niid = load_obj(path_md)[1]['Niid']


imd_losses = load_obj(path_imd)[0]

imd1_losses = load_obj(path_imd1)[0]
imd2_losses = load_obj(path_imd2)[0]
imd3_losses = load_obj(path_imd3)[0]
imd4_losses = load_obj(path_imd4)[0]
imd5_losses = load_obj(path_imd5)[0]

sigma = load_obj(path_imd)[1]['sigma']
num_particles = load_obj(path_imd)[1]['num_particles']
num_particles1 = load_obj(path_imd1)[1]['num_particles']
num_particles2 = load_obj(path_imd2)[1]['num_particles']
num_particles3 = load_obj(path_imd3)[1]['num_particles']
num_particles4 = load_obj(path_imd4)[1]['num_particles']
num_particles5 = load_obj(path_imd5)[1]['num_particles']


lr = load_obj(path_imd)[1]['lr']
decreasing_lr = load_obj(path_imd)[1]['decreasing_lr']
condition = load_obj(path_imd)[1]['condition']
d = load_obj(path_imd)[1]['d']
M = load_obj(path_imd)[1]['M']
batch_size1 = load_obj(path_imd)[1]['batch_size']
# batch_size2 = load_obj(path_imd2)[1]['batch_size']
# batch_size3 = load_obj(path_imd3)[1]['batch_size']
# batch_size4 = load_obj(path_imd4)[1]['batch_size']

path = './figures/cond_' + str(condition)
if not os.path.exists(path):
    os.mkdir(path)

path += '/d_' + str(d)

if not os.path.exists(path):
    os.mkdir(path)

path += '/Compare_IMD_NP'

if not os.path.exists(path):
    os.mkdir(path)


plt.clf()
title = 'Loss function for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)
# plt.plot(np.array(gd_losses),'k', linewidth=2, label='SGD. $ Niid = $ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(Niid, batch_size1, sigma))
# plt.plot(np.array(md_losses),'b', linewidth=0.5, label='SMD. $Niid =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(Niid, batch_size1, sigma))
plt.plot(np.array(imd_losses),'r',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles, batch_size1, sigma))
plt.plot(np.array(imd1_losses),'b',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles1, batch_size1, sigma))
plt.plot(np.array(imd2_losses),'k',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles2, batch_size1, sigma))
plt.plot(np.array(imd3_losses),'c',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles3, batch_size1, sigma))
plt.plot(np.array(imd4_losses),'m',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles4, batch_size1, sigma))
plt.plot(np.array(imd5_losses),'g',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles5, batch_size1, sigma))



# plt.plot(np.array(imd2_losses),'r',linewidth=3, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles2, batch_size2, sigma))
# plt.plot(np.array(imd3_losses),'m',linewidth=1, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles3, batch_size3, sigma))
# plt.plot(np.array(imd4_losses),'k',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles4, batch_size4, sigma))

plt.xlabel('Iterations', fontsize = 20)
plt.ylabel('Value', fontsize = 20)
plt.title(title, fontsize = 20)
plt.ylim(418,430)
plt.legend()
plt.tight_layout()
plt.savefig(path + "/SIMD_NP_M{}_sigma_{}.png".format(M, sigma), dpi=200)
plt.show()

#
plt.figure()

plt.clf()
title = 'Histogram after 100 iterations for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)

plt.hist(np.array(imd_losses)[30:], alpha=0.7, color='r',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles, batch_size1, sigma))
plt.hist(np.array(imd1_losses)[30:], alpha=0.7, color='b',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles1, batch_size1, sigma))
plt.hist(np.array(imd2_losses)[30:], alpha=0.7, color='k',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles2, batch_size1, sigma))
plt.hist(np.array(imd3_losses)[30:], alpha=0.7, color='c',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles3, batch_size1, sigma))
plt.hist(np.array(imd4_losses)[30:], alpha=0.7, color='m',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles4, batch_size1, sigma))
plt.hist(np.array(imd5_losses)[30:], alpha=0.7, color='g',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles5, batch_size1, sigma))

plt.ylabel('Frequency', fontsize = 20)
plt.title(title, fontsize = 15)
# plt.ylim(0,300)
plt.legend()
plt.tight_layout()
plt.savefig(path + "/Hist_SIMD_NP_M{}_sigma_{}.png".format(M, sigma), dpi=200)
plt.show()



# plt.clf()
# title = 'Loss function for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)
# # plt.plot(np.array(gd_losses),'r', label='SGD. $ Niid = $ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
# # plt.plot(np.array(md_losses),'b', label='SMD. $Niid =$ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
# plt.plot(np.array(imd1_losses),'b',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles1, batch_size1, sigma))
# plt.plot(np.array(imd2_losses),'r',linewidth=3, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles2, batch_size2, sigma))
# plt.plot(np.array(imd3_losses),'m',linewidth=1, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles3, batch_size3, sigma))
# plt.plot(np.array(imd4_losses),'k',linewidth=0.5, label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles4, batch_size4, sigma))
#
# plt.xlabel('Iterations', fontsize = 20)
# plt.ylabel('Value', fontsize = 20)
# plt.title(title, fontsize = 20)
# plt.ylim(259,270)
# plt.legend()
# plt.tight_layout()
# plt.savefig(path + "/IMD_M{}_sigma_{}.png".format(M, sigma), dpi=200)
# plt.show()
#
#
# plt.figure()
#
# plt.clf()
# title = 'Histogram of SIMD after 50 iterations for $n = {}$, $\kappa(W)$ = {} '.format(d, condition)
# # plt.plot(np.array(gd_losses),'r', label='SGD. $ Niid = $ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
# # plt.plot(np.array(md_losses),'b', label='SMD. $Niid =$ {:3}, $\sigma =$ {:4}'.format(Niid, sigma))
# plt.hist(np.array(imd1_losses)[150:], color='b', label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles1, batch_size1, sigma))
# plt.hist(np.array(imd3_losses)[150:], color='r', label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles3, batch_size3, sigma))
# plt.hist(np.array(imd4_losses)[150:], color='m', label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles4, batch_size4, sigma))
# plt.hist(np.array(imd2_losses)[150:], color='k', label='SIMD. $N_p =$ {:3}, $M_B$ = {}, $\sigma =$ {:4}'.format(num_particles2, batch_size2, sigma))
#
# plt.ylabel('Frequency', fontsize = 20)
# plt.title(title, fontsize = 15)
# # plt.ylim(0,300)
# plt.legend()
# plt.tight_layout()
# plt.savefig(path + "/Hist_IMD_M{}_sigma_{}.png".format(M, sigma), dpi=200)
# plt.show()
