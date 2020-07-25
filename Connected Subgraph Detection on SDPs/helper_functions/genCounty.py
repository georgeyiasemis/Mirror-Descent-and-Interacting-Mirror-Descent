import numpy as np
import sys
sys.path.insert(0, '../')
from helper_functions.load_mat import load_mat
from helper_functions.ismembertol import ismembertol, ismembertol_custom
rng = np.random.default_rng()

# xval = 1
# nSamp=10

def genCounty(xval, nSamp):

    pop_vec, yy_1, cons, S = load_mat()
    # S denotes the anomalous ground trouth cluster
    n = len(cons)

    l_0 = 5e-5
    lam_0 = pop_vec * l_0
    l_1 = xval * l_0
    lam_1 = lam_0
    lam_1[S] = pop_vec[S] * l_1

    yy_0 = rng.poisson(lam_0, (lam_0.shape[0], nSamp))
    yy_1 = rng.poisson(lam_1, (lam_1.shape[0], nSamp))

    # Adjecency matrix
    Adj = np.zeros((n, n))
    yy = np.empty((n, nSamp))
    # Position of county: Latitude and Longtitude
    fLat, fLon = np.zeros((n,1)), np.zeros((n,1))
    tol = 1e-10
    for i in range(n):
        # fix disease rates
        cons[i]['rate'] = yy_1[i,:]/cons[i]['pop']
        yy[i] = cons[i]['rate']
        cons[i]['diseased'] = 0.5 * ismembertol(np.array([i]), S)
        cons[i]['idx'] = i/n
        fLat[i] = cons[i]['Lat'][:-1].mean()
        fLon[i] = cons[i]['Lon'][:-1].mean()
        for j in range(i+1, n):
            Adj[i,j] = (ismembertol_custom(cons[i]['Lat'], cons[j]['Lat'], tol) & \
                        ismembertol_custom(cons[i]['Lon'], cons[j]['Lon'], tol)).any().astype(int)
    Adj[8,11] = 1
    Adj[11,17] = 1
    Adj += Adj.T

    # Ancor node
    s = 89

    return Adj, s, yy

# if __name__=='__main__':
    # A, s, yy = genCounty(1, 50)
    # n = A.shape[0]
    # degs = A.sum(axis=0).T
    #
    #
    # p = 2 * LA.norm(C, ord='fro') / gamma
    #
    # Y = p / degs.sum() * np.eye(n)
    # X = np.zeros((n ,n, max_iter))
    # gamma = 0.005
    # Q = paramQ_star(Adj=A, degs=degs, gamma=gamma, Y=Y, r=s)
