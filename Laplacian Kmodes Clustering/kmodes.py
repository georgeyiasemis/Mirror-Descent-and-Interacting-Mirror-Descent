import torch
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from scipy.stats import norm as normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class LaplacianKmodes():

    def __init__(self, N, D, K, lambd, var, cluster_std, random_state=1, sigma=0, num_centers=10):
        assert N >= D
        self.N = N
        self.D = D
        self.K = K
        self.lambd = lambd
        self.var = var
        self.sigma = sigma
        self.num_centers = num_centers if K <= num_centers else K
        self.stochastic = True if sigma > 0 else False
        self.X, self.labels, self.C0 = self.generate_clusters(cluster_std, random_state)
        self.C = self.C0.clone()
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.W = self.generate_affinity_matrix()


    def generate_clusters(self, cluster_std, random_state):
        '''Generates K clusters with random centers with noise'''
        X, y, centers = make_blobs(n_samples=self.N, centers=self.num_centers, n_features=self.D,
                return_centers=True, random_state=random_state, cluster_std=cluster_std)
        X = torch.tensor(X)
        y = torch.tensor(y).int()
        inds = np.random.choice(range(self.num_centers),self.K, replace=False)
        centers = torch.tensor(centers[inds])
        centers += 2 * torch.randn(centers.shape) # Add some noise

        return X, y, centers

    def generate_affinity_matrix(self):
        '''Returns a symmetric binary affinity matrix'''
        W = torch.rand(self.N, self.N).round()
        W = torch.tril(W) + torch.tril(W,-1).T

        return W


    def eval(self, Z, C_iters=100, eta_c=0.001):

        assert isinstance(Z, torch.Tensor)
        assert len(Z.shape) == 2

        # Run otpimisation for C for fixed Z with GD
        for t in range(C_iters):
            self.C -= eta_c / np.sqrt(t+1) * self.opt_C(Z, self.C)

        B = torch.zeros((self.N, self.K))
        for n in range(self.N):
            B[n] = torch.Tensor(normal.pdf(
            torch.pow((self.X[n] - self.C)/np.sqrt(self.var), 2).sum(1)))

        D = torch.diag(self.W.sum(1))
        # Laplacian
        L = D - self.W
        self.loss = (self.lambd * torch.trace(Z.T @ L @ Z ) - torch.trace(B.T @ Z)).item()
        self.gradient = self.lambd * 2 * L @ Z - B


        if self.stochastic:
            self.loss += self.sigma * torch.randn(1).item()
            self.gradient += self.sigma * torch.randn((self.K))




    def opt_C(self, Z, C):

        grad_C = torch.zeros(C.shape)
        for k in range(self.K):
            X = (self.X - C[k]) / np.sqrt(self.var)
            norm = torch.pow(X,2).sum(1)
            G_norm = normal.pdf(norm)
            grad_C[k] = (-2 / self.var * Z[:,k] * norm * G_norm) @ X
        return grad_C

    def plot_3d_clusters(self):
        with plt.style.context(('seaborn-colorblind')):
            fig = plt.figure(figsize=(20,15))
            y_unique = np.unique(self.labels)
            ax = fig.add_subplot(111, projection='3d')
            colors = np.linspace(1,10,y_unique.shape[0])
            colors = plt.cm.jet(colors/colors.max())
            for this_y in zip(y_unique):
                this_X = self.X[self.labels == this_y[0]]
                ax.scatter(this_X[:, 0], this_X[:, 1], this_X[:, 2],
                            alpha=0.4, edgecolor='k', label= str(this_y[0]),
                           s=100, color=colors[this_y[0]])
                ax.scatter(self.C0[this_y,0], self.C0[this_y,1], self.C0[this_y,2], s=200, color='k')
            ax.legend()
            ax.set_xlim(self.X[:,0].min(), self.X[:,0].max())
            ax.set_ylim(self.X[:,1].min(), self.X[:,1].max())
            ax.set_zlim(self.X[:,2].min(), self.X[:,2].max())
            plt.tight_layout()
            plt.show()
            plt.savefig('3Dgraph_N_{}_K_{}.png'.format(self.N, self.K))
if __name__ == '__main__':
    N = 3000
    D = 3
    K = 10

    C =  LaplacianKmodes(N, D, K, 1.5, 1, 2, 10 ,0)
    C.plot_3d_clusters()
    Z =  torch.ones(N,K) /K
    print(C.eval(Z))
