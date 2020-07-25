import numpy as np
import sys
sys.path.insert(0, '../')
from helper_functions.save_load import load_obj, save_obj
from startOpt.computation import computeAUC
import matplotlib.pyplot as plt

path = "../saved_data/county/MD/nSamp_50_Niid_5_Iters_50_Ng_10/"
scores_noise_path = "scores_noise"
scores_noise = load_obj(path, scores_noise_path)

gammas = np.logspace(-2,np.log10(5),10)
i = 1
for signal in ['1.1', '1.3', '1.5' , '1.7']:

    scores_signal_path = "scores_signal" + signal
    scores_signal = load_obj(path, scores_signal_path)
    aucs = computeAUC(scores_noise, scores_signal, gammas)
    save_obj(aucs, path +'aucs1.{}'.format(i))
    i += 2

s1 = load_obj(path, 'aucs1.1')
s3 = load_obj(path, 'aucs1.3')
s5 = load_obj(path, 'aucs1.5')
s7 = load_obj(path, 'aucs1.7')
for s in [s1, s3, s5, s7]:
    print('Max: {}'.format(s.max()))
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 2

plt.figure()
plt.semilogx(gammas,s1,'-o')
plt.semilogx(gammas,s3,'-x')
plt.semilogx(gammas,s5,'-d')
plt.semilogx(gammas,s7,'-+')
plt.legend(['$ \dfrac{\lambda_1}{\lambda_0}=1.1$','$\dfrac{\lambda_1}{\lambda_0}=1.3$',
            '$\dfrac{\lambda_1}{\lambda_0}=1.5$', '$\dfrac{\lambda_1}{\lambda_0}=1.7$'])
plt.legend(['$ \lambda_1/\lambda_0=1.1$','$\lambda_1/\lambda_0=1.3$',
            '$\lambda_1/\lambda_0=1.5$', '$\lambda_1/\lambda_0=1.7$'])
plt.title('  Mirror Descent. $N_{iid}=5$', fontsize=20)
plt.xlabel('$\gamma$', fontsize=20)
plt.ylabel('AUC', fontsize=20)
# plt.ylim(0.51, 0.86)
plt.tight_layout()
plt.show()
plt.savefig(path + 'aucs', dpi=500)

#
# path = "../saved_data/geo/IMD/Nnodes_nSamp_1000_Np_40_Iters_3_Ng_10/"
# scores_noise_path = "scores_noise"
# scores_noise = load_obj(path, scores_noise_path)
#
# # gammas = np.logspace(-2,np.log10(2),5)
# # i = 1
# gammas = np.array([np.sqrt( 2 / 1000)])
#
# scores_signal_path = "scores_signal"
# scores_signal = load_obj(path, scores_signal_path)
# aucs = computeAUC(scores_noise, scores_signal, gammas, 'signal')
# save_obj(aucs, path, 'aucs')
#
#
# s = load_obj(path, 'aucs')
# print('Max: {}'.format(s.max()))
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['font.size'] = '11'
# plt.rcParams['lines.linewidth'] = 2
#
# plt.figure()
# plt.semilogx(gammas,s,'-o')
# # plt.legend(['$ \dfrac{\lambda_1}{\lambda_0}=1.1$','$\dfrac{\lambda_1}{\lambda_0}=1.3$',
# #             '$\dfrac{\lambda_1}{\lambda_0}=1.5$', '$\dfrac{\lambda_1}{\lambda_0}=1.7$'])
# # plt.legend(['$ \lambda_1/\lambda_0=1.1$','$\lambda_1/\lambda_0=1.3$',
# #             '$\lambda_1/\lambda_0=1.5$', '$\lambda_1/\lambda_0=1.7$'])
# plt.title('  Interacting Mirror Descent. $N_{p}=3$', fontsize=20)
# plt.xlabel('$\gamma$', fontsize=20)
# plt.ylabel('AUC', fontsize=20)
# # plt.ylim(0.51, 0.86)
# plt.tight_layout()
# plt.show()
# plt.savefig(path + 'aucs', dpi=500)
