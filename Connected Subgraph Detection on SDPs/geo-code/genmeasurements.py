import numpy as np
import numpy.matlib
from scipy.special import gamma
import sys
sys.path.insert(0, '../')

def genMeasurements(pts, K, l_0, l_1, numSamples, thinRatio=1):

    '''
    Given pts uniform in a [-1,1]^d hypercube, generate an ellipsoid anomaly of ~K
    points centered at the origin. thinRatio gives the ratio of max semi axis
    length to min semi axis length. Non-anomalous are Poisson(l_0), anomalous
    are Poisson(l_1) with numSamples IID copies.
    '''
    n, d = pts.shape

    numLargerDims = 1;

    while numLargerDims < d:

        r = 2 / np.sqrt(np.pi) * \
            (K/n / thinRatio**numLargerDims * gamma(d/2 + 1)) ** (1/d)

        if r * thinRatio <= 1:
            radii = r * np.concatenate((np.ones((d - numLargerDims)),
                thinRatio * np.ones((numLargerDims))))
            break
        numLargerDims += 1

    dists = ((pts / np.matlib.repmat(radii, n, 1)) ** 2).sum(axis=1)

    S_ind = dists <= 1
    S = np.where(S_ind)[0]
    k = S.shape[0]
    yy = np.zeros((n, numSamples))
    yy[np.logical_not(S_ind), :] = np.random.poisson(l_0, (n-k, numSamples))
    yy[S_ind, :] = np.random.poisson(l_1, (k, numSamples))

    return yy, S
#
# if __name__ == "__main__":
#     # pts = np.ones((10,3))
#     # pts[7,1] = 0.5
#     # pts[3,0] = 5
#     # pts[9,2] = 9
#     # pts = pts/20
#     # K = 2
#     # l_0 = 1
#     # l_1 = 3
#     # numSamples = 50
#     # yy, S = genMeasurements(pts, K, l_0, l_1, numSamples, 0.1)
#     # print(yy)
#     # print(S)
