import torch
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_blobs, make_moons
from scipy.stats import norm as normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def one_hot(a, num_classes):

    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class LaplacianKmodes():

    def __init__(self, N, D, K, lambd, var, cluster_std, random_state=1, sigma=0, mode='blobs'):
        assert N >= D
        self.N = N
        self.D = D
        self.K = K
        self.lambd = lambd
        self.var = var
        self.sigma = sigma
        self.stochastic = True if sigma > 0 else False
        if mode == 'blobs':
            self.X, self.labels, self.C = self.generate_clusters(cluster_std, random_state)
        elif mode == 'moons':
            self.K = 2
            self.D = 2
            self.X, self.labels, self.C = self.generate_moons(cluster_std, random_state)

        self.C = self.C.clone()
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.W = self.generate_affinity_matrix()


    def generate_clusters(self, cluster_std, random_state):
        '''Generates K clusters with random centers with noise'''
        X, y, C = make_blobs(n_samples=self.N, centers=self.K, n_features=self.D,
                return_centers=True, random_state=random_state, cluster_std=cluster_std)
        X = torch.tensor(X)
        y = torch.tensor(y).int()
        self.true_centers = C
        centers = np.random.uniform(low=tuple(X.min(axis=0)[0]), high=tuple(X[:].max(axis=0)[0]), size=(self.K, self.D))
        centers = torch.tensor(centers).float()

        return X, y, centers

    def generate_moons(self, cluster_std, random_state):
        assert self.K == 2
        assert self.D == 2
        X, y = make_moons(n_samples=self.N, noise=cluster_std, random_state=random_state)
        X = torch.tensor(X)

        y = torch.tensor(y).int()

        centers = np.random.uniform(low=(X[:, 0].min(), X[:, 1].min()), high=(X[:, 0].max(), X[:, 1].max()), size=(2,2))
        centers = torch.tensor(centers).float()
        return X, y, centers

    def generate_affinity_matrix(self, exp=True):
        '''Returns a symmetric binary affinity matrix'''
        if not exp:
            W = torch.rand(self.N, self.N).round()
            W = torch.tril(W) + torch.tril(W,-1).T

        else:
            gamma = 0.5
            X1 = self.X.clone().reshape(self.N, 1, self.D)
            X2 = self.X.clone().reshape(1, self.N, self.D)
            W = torch.exp(-gamma * torch.sum(torch.pow(X1-X2, 2), axis=-1)).float()
            print(W)
        return W


    def eval(self, Z, C_iters=100, eta_c=0.00001):

        assert isinstance(Z, torch.Tensor)
        assert len(Z.shape) == 2
        self.Z = Z

        B = torch.zeros((self.N, self.K))

        for n in range(self.N):
            B[n] = torch.Tensor(normal.pdf(
            torch.pow((self.X[n] - self.C)/np.sqrt(self.var), 2).sum(1)))

        D = torch.diag(self.W.sum(1))
        # Laplacian
        L = D - self.W
        self.loss = (self.lambd * torch.trace(Z.T @ L @ Z ) - torch.trace(B.T @ Z)).item()
        self.gradient = self.lambd * 2 * L @ Z - B
        self.accuracy = self.eval_accuracy()
        self.auc = self.eval_auc_score()

        # Run otpimisation for C for fixed Z with GD
        for t in range(C_iters):
            self.C -= eta_c / np.sqrt(t + 1) * self.opt_C(Z, self.C)

        if self.stochastic:
            self.loss += self.sigma * torch.randn(1).item()
            self.gradient += self.sigma * torch.randn((self.K))


    def eval_accuracy(self):

        true = self.labels
        pred = self.Z.argmax(axis=1).int()

        return sum(true == pred).item() / self.N

    def eval_auc_score(self):

        if self.K > 2:
            y = self.labels.numpy()
            true = one_hot(y, self.K)
            # print(true)
            true_scores = self.Z.numpy()
        else:
            true = self.labels.numpy()
            true_scores = self.Z[:,1].numpy()

        return roc_auc_score(true, true_scores)

    def opt_C(self, Z, C):

        grad_C = torch.zeros(C.shape)
        for k in range(self.K):
            X = (self.X - C[k]) / np.sqrt(self.var)
            norm = torch.pow(X,2).sum(1)
            G_norm = normal.pdf(norm)
            grad_C[k] = (-2 / self.var * Z[:,k] * norm * G_norm) @ X
        return grad_C

    def plot_2d_clusters(self, y=None, C=None):

        assert self.D == 2
        if y == None:
            y = self.labels
        if C == None:
            C = self.C

        with plt.style.context(('seaborn-colorblind')):
            # fig = plt.figure(figsize=(20,15))
            y_unique = np.unique(y)
            colors = plt.cm.jet(y_unique*100)
            for this_y in zip(y_unique):
                this_X = self.X[y  == this_y[0]]

                plt.scatter(this_X[:, 0], this_X[:, 1], alpha=0.4, edgecolor='k', label= str(this_y[0]),
                        s=100, color=colors[this_y[0]])
                plt.scatter(C[this_y,0], C[this_y,1], s=200, color='k')
            plt.legend()
            # plt.xlim(self.X[:,0].min(), self.X[:,0].max())
            # plt.ylim(self.X[:,1].min(), self.X[:,1].max())
            plt.tight_layout()
            plt.show()
            plt.savefig('2Dgraph_N_{}_K_{}.png'.format(self.N, self.K))

    def plot_3d_clusters(self, y=None, C=None):

        assert self.D == 3
        if y == None:
            y = self.labels
        if C == None:
            C = self.C

        with plt.style.context(('seaborn-colorblind')):
            fig = plt.figure(figsize=(20,15))
            y_unique = np.unique(y)
            ax = fig.add_subplot(111, projection='3d')
            colors = np.linspace(1,10,y_unique.shape[0])
            colors = plt.cm.jet(colors/colors.max())
            for this_y in zip(y_unique):
                this_X = self.X[y == this_y[0]]
                ax.scatter(this_X[:, 0], this_X[:, 1], this_X[:, 2], alpha=0.4, edgecolor='k', label= str(this_y[0]),
                           s=100, color=colors[this_y[0]])
                ax.scatter(C[this_y,0], C[this_y,1], C[this_y,2], s=200, color='k')
            ax.legend()
            ax.set_xlim(self.X[:,0].min(), self.X[:,0].max())
            ax.set_ylim(self.X[:,1].min(), self.X[:,1].max())
            ax.set_zlim(self.X[:,2].min(), self.X[:,2].max())
            plt.tight_layout()
            plt.show()
            plt.savefig('3Dgraph_N_{}_K_{}.png'.format(self.N, self.K))

if __name__ == '__main__':
    N = 1000
    D = 2
    K = 2

    C =  LaplacianKmodes(N, D, K, 1, 1, 0.2, mode='moons')
    C.plot_2d_clusters()
    Z =  torch.ones(N,K) /K
    print(C.eval(Z))
