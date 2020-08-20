import numpy as np
# import spicy as sps
# from geo_code.helper_functions.ind2sub import ind2sub

def Q_gamma_M(Adj, u, gamma, r):
    """Computes
        Q_{\gamma}(M) = sum_{ij in E} M_ij L_ij - \gamma sum_{i in V} M_ii L_ri
    where M \ approx u*u' with n x 1 vector u

    Parameters
    ----------
    Adj : np.array
        Adjacency matrix of graph of shape (n, n).
    u : np.array
        Vector of shape (n,1).
    gamma : float
        Gamma parameter.
    r : int
        Root vertex.

    Returns
    -------
    Q
        Description of returned object.

    """

    n = Adj.shape[0]
    degs = np.array(Adj.sum(0)).reshape(-1,1)

    X = u * u.T
    Q = np.diag(np.sum(X * Adj, 0)) - X * Adj

    dX = degs.squeeze() * np.diag(X).squeeze()
    starX = np.diag(dX)
    starX[r,:] = - dX
    starX[:,r] = - dX
    starX[r,r] = dX.sum() - degs[r] * X[r,r]

    Q = Q - gamma * starX

    return Q


def paramQ_star(Adj, degs, gamma, r, X=None, Y=None):

    if type(Y) == np.ndarray:
        n = Y.shape[0]
        i, j = np.where(np.triu(Adj))
        yvals = np.zeros((len(i), 1))

        for (c ,(w, v)) in enumerate(zip(i, j)):
            yvals[c] = Y[i[c], i[c]] + Y[j[c], j[c]] - Y[j[c], i[c]] - Y[i[c], j[c]]

        YY = np.zeros((n,n))
        YY[i, j] = yvals.reshape(-1)
        YY = YY + YY.T

        dummy = np.diag(degs * (Y[r,r] + np.diag(Y) - Y[r,:].T - Y[:,r]))

        out = YY - gamma * dummy

        return out

    elif type(X) == np.ndarray:

        Q = np.diag((X * Adj).sum(0)) - X * Adj
        dX = degs * np.diagonal(X)
        starX = np.diag(dX)

        starX[r,:] = -dX
        starX[:,r] = -dX
        starX[r,r] = dX.sum() - degs[r] * X[r,r]

        out = Q - gamma * starX
        return out
