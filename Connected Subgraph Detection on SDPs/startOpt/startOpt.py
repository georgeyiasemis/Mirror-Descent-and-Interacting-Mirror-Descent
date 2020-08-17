import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import sys
from tqdm import tqdm,  trange
sys.path.insert(0, '../')


from startOpt.q_gamma import  paramQ_star

def startOpt_md(A, C, gamma, s, max_iter=100):

    n = A.shape[0]
    degs = A.sum(axis=0).T

    # Learning Rate
    eta = 5

    p = 2 * LA.norm(C, ord='fro') / gamma

    tic()

    Y = p / degs.sum() * np.eye(n)
    X = np.zeros((n ,n, max_iter))

    grad = np.zeros(n)

    v0 = np.ones(n)

    for i in range(max_iter):

        Q = paramQ_star(Adj=A, degs=degs, gamma=gamma, Y=Y, r=s)
        _, u = eigsh(np.real(Q + C), k=1, which='LA', v0=v0, tol=0.000001)
        v0 = u
        X[:,:,i] = u * u.T
        QX = paramQ_star(A, degs, gamma, r=s, X=X[:,:,i])

        # Y = expm(eta * grad / degs.max())
        grad = QX
        Y = Y * expm( - eta * grad / degs.max() )

        Y = Y / np.trace(np.diag(degs) * Y) * p

    M = X.mean(2)

    return M

def startOpt_imd(A, C, gamma, s, max_iter=50, num_particles=10):
    n = A.shape[0]
    degs = A.sum(axis=0).T

    # Learning Rate
    eta = 5
    p = 2 * LA.norm(C, ord='fro') / gamma
    D = degs.max()

    # A_ij = 1/Np
    interaction_mat = np.ones((num_particles, num_particles)) / num_particles
    particles = np.empty((n,n,num_particles), dtype=np.float)

    for j in range(num_particles):
        P =  p / degs.sum() * (np.eye(n) + np.random.rand(n,n) * 0.1)
        P = (P + P.T) / 2
        particles[:, : ,j] = P / np.trace(P) * p

    X = np.zeros((n ,n, max_iter, num_particles))


    for i in range(max_iter):
        grad = np.zeros(n)
        v0 = np.ones(n)
        for part in range(num_particles):

            Y = particles[:,:,part].copy()

            Q = paramQ_star(Adj=A, degs=degs, gamma=gamma, Y=Y, r=s)
            _, u = eigsh(np.real(Q + C), k=1, which='LA', v0=v0, tol=0.000001)
            v0 = u
            X[:, :, i, part] = u.reshape(-1,1) * u.reshape(1,-1) #/ num_particles
            QX = paramQ_star(A, degs, gamma, r=s, X=X[:,:,i, part].squeeze())


            grad = QX
            # particle_mean = np.average(particles, axis=2, weights=interaction_mat[part,:])
            #
            # particle_mean += particles[:,:,part] * (1 - interaction_mat[part,part])
            # particle_mean = particle_mean.reshape(n,n)
            # Y = particle_mean * expm( - eta * grad / D)

            # # Y_i' = Prod{i=1}^N Particles_{ij}^A_{ij}
            Y = np.prod(np.power(particles, interaction_mat[part]), axis=2)


            # Y_i = Y_i ' exp(-eta * grad_Y)
            Y = Y * expm( - eta * grad)
            # Normalise trace to p
            Y = Y / np.trace(np.diag(degs) * Y) * p

    M = X.mean((2,3))
    return M

def startOpt(A, C, gamma, s, max_iter=100):
    losses = []
    n = A.shape[0]
    degs = A.sum(axis=0).T

    # Learning Rate
    eta = 5
    p = 2 * LA.norm(C, ord='fro') / gamma

    tic()

    Y = p / degs.sum() * np.eye(n)
    X = np.zeros((n ,n, max_iter))

    grad = np.zeros(n)

    v0 = np.ones(n)
    for i in range(max_iter):

        Q = paramQ_star(Adj=A, degs=degs, gamma=gamma, Y=Y, r=s)
        if i >= 1:
            loss = np.trace(np.real(Q + C) @ M)
            losses.append(loss)

        _, u = eigsh(np.real(Q + C), k=1, which='LA', v0=v0, tol=0.000001)
        v0 = u
        X[:,:,i] = u * u.T
        QX = paramQ_star(A, degs, gamma, r=s, X=X[:,:,i])


        grad = grad - QX
        Y = expm(eta * grad / degs.max())

        Y = Y / np.trace(np.diag(degs) * Y) * p

        M = X.mean(2)
    # M = X.mean(2)
    # print('Time elapsed:', toc(), 'seconds.')
    # print(losses[-1])
    return M, losses

if __name__ == '__main__':
    A = np.eye(10)
    A[1,3] = 1
    A[3,1] = 1
    A[5,2] = 1
    A[2,5] = 1
    A[9,3] = 1
    A[3,9] = 1
    C = np.array([[  46/5, 99/10,  1/10,   4/5,   3/2, 67/10,  37/5, 51/10,  29/5,     4],
        [  49/5,     8,  7/10,   7/5,   8/5, 73/10,  11/2, 57/10,  32/5, 41/10],
        [   2/5, 81/10,  44/5,     2,  11/5,  27/5,  28/5, 63/10,     7, 47/10],
        [  17/2, 87/10, 19/10, 21/10,  3/10,     6,  31/5, 69/10, 71/10,  14/5],
        [  43/5, 93/10,   5/2,   1/5,  9/10, 61/10,  34/5,  15/2,  26/5,  17/5],
        [ 17/10,  12/5,  38/5, 83/10,     9,  21/5, 49/10,  13/5, 33/10,  13/2],
        [ 23/10,   1/2,  41/5, 89/10, 91/10,  24/5,     3,  16/5, 39/10,  33/5],
        [ 79/10,   3/5, 13/10,  19/2, 97/10, 29/10, 31/10,  19/5,   9/2,  36/5],
        [     1,   6/5,  47/5,  48/5,  39/5,   7/2, 37/10,  22/5,  23/5, 53/10],
        [ 11/10,   9/5,    10, 77/10,  42/5,  18/5, 43/10,     5, 27/10, 59/10]])
    gamma=1
    S=3

    M=startOpt(A, C, gamma, S)
    print(M.shape)
    print(M)

    print(M.max(), M.min())
