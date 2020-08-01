import matplotlib.pyplot as plt
import numpy as np
from helper_functions.save_load import load_obj
import os
from kmodes import LaplacianKmodes
import torch



plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 0.2
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


path = "./saved_items/N_100_D_2_K_2_lambda_1_var_5_clusterstd_0.2/"

path_sgd = path + "GD_Niid_5_lr_0.0005_iters_10"
gd_kmodes = torch.load(path_sgd + '_model.pt')

path_smd = path + "MD_Niid_5_lr_0.0001_iters_10"
md_kmodes = torch.load(path_smd + '_model.pt')

path_imd = path + "IMD_Np_5_lr_0.0001_iters_10"
imd_kmodes = torch.load(path_imd + '_model.pt')

sgd_losses, args_gd = load_obj(path_sgd)
smd_losses, args_md = load_obj(path_smd)
imd_losses, args_imd = load_obj(path_imd)



print((gd_kmodes.Z.argmax(axis=1) == gd_kmodes.labels.squeeze()).sum())
print((md_kmodes.Z.argmax(axis=1) == md_kmodes.labels.squeeze()).sum())
print((imd_kmodes.Z.argmax(axis=1) == imd_kmodes.labels.squeeze()).sum())

y = imd_kmodes.Z.argmax(axis=1)
imd_kmodes.plot_2d_clusters(y, C=imd_kmodes.C)
#
#
# # print(str(args))
# path = './figures/'
# if not os.path.exists(path):
#     os.mkdir(path)
# #
# plt.figure()
# plt.plot(np.array(smd_losses['accuracy'])[:],linewidth=1, label='MD')
# plt.plot(np.array(sgd_losses['accuracy'])[:],linewidth=1, label='GD')
# plt.plot(np.array(imd_losses['accuracy'])[:],linewidth=1, label='IMD')
# # # plt.semilogy(np.array(simd_losses50),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(50, sigma))
# # # plt.semilogy(np.array(simd_losses100),linewidth=1, label='SIMD. $ Np = $ {:3}, $\sigma =$ {:4}'.format(100, sigma))
# # plt.ylabel('Function Value')
# # plt.xlabel('Iterations')
# # title = 'Loss function for $N$ = {}, $K$ = {}, $D$ = {}'.format(args['N'], args['K'], args['D'])
# # plt.title(title, fontsize=20)
# # plt.tight_layout()
# plt.legend()
# # plt.show()
# # plt.ylim(min(smd_losses)-2, 10)
# plt.savefig(path + 'fig.png', dpi=300)
