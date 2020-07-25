import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')

from geo_code.helper_functions.save_load import load_obj, save_obj

#
path = "../aistats-code/saved_data/"
# scores_noise_path = "scores_noise"
path_losses = "scores_noise_losses"
# scores_noise_path = "scores_noise"
# scores_noise = load_obj(path, scores_noise_path)
losses = load_obj(path, path_losses)
# gammas = np.logspace(-2,np.log10(5),10)


# scores_noise = scores_noise.mean(1)
print(losses.shape)
print(losses[:,0,:].shape)
print(losses[:,0,:].mean(0).shape)
for i in range(losses.shape[1]):
    plt.plot(losses[:,i,:].mean(0))
plt.savefig('fig.png')

# print(scores_noise)
