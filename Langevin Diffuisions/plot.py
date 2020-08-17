import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '11'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

iters_NLA = torch.load('./saved_items/d_100/iters_NLA.pkl')[0]
args_NLA = torch.load('./saved_items/d_100/iters_NLA.pkl')[1]
print(args_NLA)
iters_INLA = torch.load('./saved_items/d_100/iters_INLA.pkl')[0]
args_INLA = torch.load('./saved_items/d_100/iters_INLA.pkl')[1]
print(args_INLA)

plt.figure()
plt.plot(iters_NLA['mean_error'], label=r'NLA, $\epsilon$ = 0.7, $N_{iid}$ = 10')
plt.plot(iters_INLA['mean_error'], label=r'INLA, $\epsilon$ = 0.7,  $N_{p}$ = 10')
plt.title(r'$log_{10}||\, \bar{X_t}\,||_2^2$', fontsize=20)
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Function Value', fontsize=20)
plt.legend()
plt.savefig('./figures/d_100/loss.png')

plt.figure()
plt.plot(iters_NLA['scatter_error'], label=r'NLA, $\epsilon$ = 0.7,  $N_{iid}$ = 10')
plt.plot(iters_INLA['scatter_error'], label='INLA, $\epsilon$ = 0.7,  $N_p$ = 10')
plt.title(r'$log_{10}\frac{|| \,\hat{\Sigma}\,-\,\Sigma\,||_F^2}{||\,\Sigma\,||_F^2}$', fontsize=20)
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Function Value', fontsize=20)
plt.legend()
plt.savefig('./figures/d_100/scatter_loss.png')

print('Variance of NLA: {}'.format(np.array(iters_NLA['scatter_error'])[500:].var()))
print('Variance of INLA: {}'.format(np.array(iters_INLA['scatter_error'])[500:].var()))
plt.figure()
plt.hist(np.array(iters_NLA['scatter_error'])[500:], label=r'NLA, $\epsilon$ = 0.7,  $N_{iid}$ = 10')
plt.hist(np.array(iters_INLA['scatter_error'])[500:], label='INLA, $\epsilon$ = 0.7,  $N_p$ = 10')
plt.title(r'$log_{10}\frac{||\, \hat{\Sigma}\,-\,\Sigma\,||_F^2}{||\,\Sigma\,||_F^2}$ after 500 iterations', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.legend()
plt.savefig('./figures/d_100/hist_scatter_loss.png')
