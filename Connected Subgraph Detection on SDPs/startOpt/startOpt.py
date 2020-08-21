import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import sys
from tqdm import tqdm,  trange
sys.path.insert(0, '../')
from startOpt.q_gamma import  paramQ_star

def runOpt_md(A, C, gamma, s, max_iter=100):

    n = A.shape[0]
    degs = A.sum(axis=0).T

    # Learning Rate
    eta = 5

    p = 2 * LA.norm(C, ord='fro') / gamma

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

def runOpt_imd(A, C, gamma, s, max_iter=50, num_particles=10):
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

            particles[:, : ,part] = Y.copy()

    M = X.mean((2,3))
    return M
