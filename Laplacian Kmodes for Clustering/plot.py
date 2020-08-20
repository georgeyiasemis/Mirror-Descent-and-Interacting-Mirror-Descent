import matplotlib.pyplot as plt
import numpy as np
from helper_functions.save_load import load_obj
import os
from kmodes import LaplacianKmodes
import torch

'''
Plot saved results.
'''


plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.2
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


path = "./saved_items/N_1000_D_2_K_2_lambda_2_var_0.65_clusterstd_0.15_moons/"
#
path_sgd = path + "GD_Niid_2_lr_1e-05_iters_500"
gd_kmodes = torch.load(path_sgd + '_model.pt')

path_smd = path + "MD_Niid_2_lr_1e-05_iters_500"
md_kmodes = torch.load(path_smd + '_model.pt')

path_imd = path + "IMD_Np_2_lr_1e-05_iters_500"
imd_kmodes = torch.load(path_imd + '_model.pt')

smd_losses, args_md = load_obj(path_smd)
sgd_losses, args_gd = load_obj(path_sgd)
imd_losses, args_imd = load_obj(path_imd)



print('GD Accuracy: {}. GD AUC: {}.'.format(sgd_losses['accuracy'][-1], sgd_losses['auc'][-1]))
y = gd_kmodes.Z.argmax(axis=1)
plt.figure()
imd_kmodes.plot_2d_clusters(path +'GD', title='GD, Niid = {}'.format(args_gd['Niid']),
                y=y, C=gd_kmodes.C)

print('MD Accuracy: {}. MD AUC: {}.'.format(smd_losses['accuracy'][-1], smd_losses['auc'][-1]))
y = md_kmodes.Z.argmax(axis=1)
plt.figure()
imd_kmodes.plot_2d_clusters(path +'MD', title='MD, Niid = {}'.format(args_md['Niid']),
                y=y, C=md_kmodes.C)

print('IMD Accuracy: {}. IMD AUC: {}.'.format(imd_losses['accuracy'][-1], imd_losses['auc'][-1]))
y = imd_kmodes.Z.argmax(axis=1)
plt.figure()
plt.title('Results for IMD')
imd_kmodes.plot_2d_clusters(path +'IMD' ,  title='IMD, N_p = {}'.format(args_imd['num_particles']),
                y=y, C=imd_kmodes.C)
# plt.figure()
# imd_kmodes.plot_2d_clusters(path +'original', title='Two half-moons, N = {}'.format(args_md['N']))




plt.figure()
plt.plot(np.array(smd_losses['loss'])[:],linewidth=1, label='MD. $Niid$ = {}'.format(args_md['Niid']))
plt.plot(np.array(sgd_losses['loss'])[:],linewidth=1, label='GD, $Niid$ = {}'.format(args_gd['Niid']))
plt.plot(np.array(imd_losses['loss'])[:],linewidth=1, label='IMD, $N_p$ = {}'.format(args_imd['num_particles']))
plt.ylabel('Function Value')
plt.xlabel('Iterations')
title = 'Loss function for $N$ = {}, $K$ = {}, $D$ = {}'.format(args_imd['N'], args_imd['K'], args_imd['D'])

plt.title(title, fontsize=15)
# plt.tight_layout()
plt.legend()
plt.show()
plt.ylim(min(imd_losses['loss'])-0.015, max(imd_losses['loss'])+0.015)
plt.savefig(path + 'losses.png')
plt.figure()
plt.hist(np.array(smd_losses['loss'])[300:],linewidth=1, label='MD. $Niid$ = {}'.format(args_md['Niid']))
plt.hist(np.array(sgd_losses['loss'])[300:],linewidth=1, label='GD, $Niid$ = {}'.format(args_gd['Niid']))
plt.hist(np.array(imd_losses['loss'])[300:],linewidth=1, label='IMD, $N_p$ = {}'.format(args_imd['num_particles']))
plt.xlabel('Function Value')
title = 'Histogram of loss function for $N$ = {}, $K$ = {}, $D$ = {}'.format(args_imd['N'], args_imd['K'], args_imd['D'])
plt.title(title, fontsize=15)
# plt.tight_layout()
plt.legend()
plt.show()
plt.savefig(path + 'hist.png', dpi=300)
