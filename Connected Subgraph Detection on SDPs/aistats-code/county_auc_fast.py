import numpy as np
import sys
sys.path.insert(0, '../')

from geo_code.genCounty import genCounty
from geo_code.helper_functions.save_load import save_obj
from startOpt.startOpt_fast import startOpt_fast
from Time.tictoc import tic, toc

def county_auc_fast(gammas_n, signal, nSamp=50):
    gammas = np.logspace(-2, np.log10(5), gammas_n)
    times = []

    A, s, yy = genCounty(signal, nSamp)
    scores_noise = np.zeros((nSamp, 10))

    tic()
    for ns in range(nSamp):
        ys = yy[:, ns]
        c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])

        for gind in range(10):
            u_s = startOpt_fast(A, c.squeeze(), gammas[gind], s, 20, 100)
            scores_noise[ns, gind] = ((ys @ u_s) ** 2).mean()

    time = toc()
    times.append(time)
    print()
    if signal == 1:
        save_obj(scores_noise, '../aistats-code/saved_data/scores_noise_fast')
        print('Noise fast done in', time, 'seconds.')
    else:
        save_obj(scores_noise, '../aistats-code/saved_data/scores_signal_fast', str(signal))
        print('Signal fast' + str(signal) +' done in', time, 'seconds.')
    print()

    save_obj(times, '../aistats-code/saved_data/times_fast')

if __name__ == '__main__':
    signals = [1, 1.5, 1.1, 1.3]
    for signal in signals:
        county_auc_fast(10, signal)
    #
    #
    #
    # A, s, yy = genCounty(1.5, nSamp)
    #
    # scores_signal15 = np.zeros((nSamp, 10))
    # tic()
    # for ns in range(nSamp):
    #     ys = yy[:,ns]
    #     c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])
    #     C = c.reshape(-1,1) @ c.reshape(1,-1)
    #
    #     for gind in range(10):
    #         M = startOpt(A, C, gammas[gind], s)
    #         scores_signal15[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
    #
    # save_obj(scores_signal15, 'aistats-code/saved_data/scores_signal1_5')
    # toc = toc()
    # times.append(toc)
    # print()
    # print('Signal 1.5 done in', toc, 'seconds.')
    # print()
    # ##---------------------------------------------
    # ##---------------------------------------------
    #
    # A, s, yy = genCounty(1.3, nSamp)
    #
    # scores_signal13 = np.zeros((nSamp, 10))
    # tic()
    # for ns in range(nSamp):
    #     ys = yy[:,ns]
    #     c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])
    #     C = c.reshape(-1,1) @ c.reshape(1,-1)
    #
    #     for gind in range(10):
    #         M = startOpt(A, C, gammas[gind], s)
    #         scores_signal13[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
    #
    #
    # save_obj(scores_signal13, 'aistats-code/saved_data/scores_signal1_3')
    # toc = toc()
    # times.append(toc)
    # print()
    # print('Signal 1.3 done in', toc, 'seconds.')
    # print()
    # ##---------------------------------------------
    # ##---------------------------------------------
    #
    # A, s, yy = genCounty(1.1, nSamp)
    #
    # scores_signal11 = np.zeros((nSamp, 10))
    # tic()
    # for ns in range(nSamp):
    #     ys = yy[:,ns]
    #     c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])
    #     C = c.reshape(-1,1) @ c.reshape(1,-1)
    #
    #     for gind in range(10):
    #         M = startOpt(A, C, gammas[gind], s)
    #         scores_signal11[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
    #
    # save_obj(scores_signal11, 'aistats-code/saved_data/scores_signal1_1')
    # toc = toc()
    # times.append(toc)
    # print()
    # print('Signal 1.1 done in', toc, 'seconds.')
    # print()
    # ##---------------------------------------------
    # save_obj(times, 'aistats-code/saved_data/times')

    ##---------------------------------------------
    #
    #
    # for val in calc:
    #     A, s, yy = genCounty(1.1, nSamp)
    #
    #     scores_signal11 = np.zeros((nSamp, 10))
    #     tic()
    #     for ns in range(nSamp):
    #         ys = yy[:,ns]
    #         c = ys / np.linalg.norm(ys) * np.sqrt(ys.shape[0])
    #         C = c.reshape(-1,1) @ c.reshape(1,-1)
    #
    #         for gind in range(10):
    #             M = startOpt(A, C, gammas[gind], s)
    #             scores_signal11[ns, gind] = np.trace(ys.reshape(-1,1) @ ys.reshape(1,-1) @ M)
    #
    #     save_obj(scores_signal11, 'aistats-code/saved_data/scores_signal1_1')
    #     toc = toc()
    #     times.append(toc)
    #     print()
    #     print('Signal 1.1 done in', toc, 'seconds.')
    #     print()
    #
