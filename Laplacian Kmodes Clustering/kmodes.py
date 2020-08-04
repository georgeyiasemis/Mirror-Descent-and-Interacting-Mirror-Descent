import torch
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_blobs, make_moons
from scipy.stats import norm as normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import MeanShift, estimate_bandwidth


def one_hot(a, num_classes):

    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class LaplacianKmodes():

    def __init__(self, N, D, K, lambd, bandwidth, cluster_std, random_state=1, sigma=0, mode='blobs'):
        assert N >= D
        self.N = N
        self.D = D
        self.K = K
        self.lambd = lambd
        self.bandwidth = bandwidth
        self.sigma = sigma
        self.stochastic = True if sigma > 0 else False
        if mode == 'blobs':
            self.X, self.labels, self.C = self.generate_clusters(cluster_std, random_state)
        elif mode == 'moons':
            self.K = 2
            self.D = 2
            self.X, self.labels, self.C = self.generate_moons(cluster_std, random_state)

        self.C0 = self.C.clone()
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

        # centers = np.random.uniform(low=(X[:, 0].min(), X[:, 1].min()), high=(X[:, 0].max(), X[:, 1].max()), size=(2,2))

        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True)
        ms.fit(X.numpy())
        cluster_centers = ms.cluster_centers_
        centers = torch.tensor(cluster_centers)
        centers = torch.tensor(centers).float()
        return X, y, centers

    def generate_affinity_matrix(self, bin=False):
        '''Returns a symmetric binary affinity matrix'''
        if bin:
            W = kneighbors_graph(self.X, int(self.N / self.K), mode='connectivity', include_self=True)
            W = torch.tensor(W.toarray()).float()
        else:
            gamma = 0.5
            X1 = self.X.clone().reshape(self.N, 1, self.D)
            X2 = self.X.clone().reshape(1, self.N, self.D)
            dist = torch.sum(torch.pow(X1-X2, 2), axis=-1)
            W = torch.exp(-gamma * dist).float()
            print(W)
        return W


    def eval(self, Z, C_iters=10, eta_c=0.005):

        assert isinstance(Z, torch.Tensor)
        assert len(Z.shape) == 2
        self.Z = Z
        # Run otpimisation for C for fixed Z with GD
        for t in range(C_iters):
            C = self.C.clone()
            self.C = C - eta_c  * self.eval_grad_C(Z, C)


        B = torch.zeros((self.N, self.K))

        for n in range(self.N):
            B[n] = torch.Tensor(normal.pdf(
            torch.pow((self.X[n] - self.C)/np.sqrt(self.bandwidth), 2).sum(1)))

        D = torch.diag(self.W.sum(1))
        # Laplacian
        L = D - self.W
        self.loss = (self.lambd * torch.trace(Z.T @ L @ Z ) - torch.trace(B.T @ Z)).item()
        self.gradient = self.lambd * 2 * L @ Z - B
        self.accuracy = self.eval_accuracy()
        self.auc = self.eval_auc_score()



        if self.stochastic:
            self.loss += self.sigma * torch.randn(1).item()
            self.gradient += self.sigma * torch.randn((self.K))

    def eval_grad_C(self, Z, C):

        grad_C = torch.zeros(C.shape)
        for k in range(self.K):
            X = (self.X - C[k]) / self.bandwidth
            norm = torch.pow(X,2).sum(1)
            G_norm = normal.pdf(norm)
            grad_C[k] = (-2 * Z[:,k] * norm * G_norm) @ X / self.bandwidth
        return grad_C + self.sigma * torch.randn((self.C.shape))
    def eval_accuracy(self):

        true = self.labels
        pred = self.Z.argmax(axis=1).int()

        return sum(true == pred).item() / self.N

    def eval_auc_score(self):

        if self.K > 2:
            y = self.labels.numpy()
            true = one_hot(y, self.K)
            true_scores = self.Z.numpy()
        else:
            true = self.labels.numpy()
            true_scores = self.Z[:,1].numpy()

        return roc_auc_score(true, true_scores)

    def plot_2d_clusters(self, path='', title='', y=None, C=None):

        assert self.D == 2
        if y == None:
            y = self.labels
        if C == None:
            C = self.C0

        with plt.style.context(('seaborn-colorblind')):

            y_unique = np.unique(y)
            colors = np.linspace(0,1,self.K)
            colors = plt.cm.jet(colors)
            for this_y in y_unique:
                this_X = self.X[y  == this_y]

                plt.scatter(this_X[:, 0], this_X[:, 1], alpha=0.4, edgecolor='k', label= str(this_y),
                        s=100, color=colors[this_y])
                plt.scatter(C[this_y,0], C[this_y,1], s=200, color='k')
            plt.legend()
            plt.title(title, fontsize=15)
            plt.tight_layout()
            plt.show()
            plt.savefig(path + '2Dgraph_N_{}_K_{}.png'.format(self.N, self.K))

    def plot_3d_clusters(self, path='', y=None, C=None):

        assert self.D == 3
        if y == None:
            y = self.labels
        if C == None:
            C = self.true_centers

        with plt.style.context(('seaborn-colorblind')):
            fig = plt.figure(figsize=(20,15))
            y_unique = np.unique(y)
            ax = fig.add_subplot(111, projection='3d')
            colors = np.linspace(0,1,self.K)
            colors = plt.cm.jet(colors)
            for this_y in y_unique:
                color = colors[this_y]
                # print(this_y)
                this_X = self.X[y == this_y]
                ax.scatter(this_X[:, 0], this_X[:, 1], this_X[:, 2], alpha=0.15, s=200, edgecolor='k', label= str(this_y),
                            color=color)
                ax.scatter(C[this_y,0], C[this_y,1], C[this_y,2], s=300, color='k')
            ax.legend(fontsize=15)
            ax.set_xlim(self.X[:,0].min(), self.X[:,0].max())
            ax.set_ylim(self.X[:,1].min(), self.X[:,1].max())
            ax.set_zlim(self.X[:,2].min(), self.X[:,2].max())
            plt.title('$N$ = {}, $K$ = {}'.format(self.N, self.K), fontsize=40)
            plt.tight_layout()
            plt.show()
            plt.savefig(path + '3Dgraph_N_{}_K_{}.png'.format(self.N, self.K))

if __name__ == '__main__':
    N = 1000
    D = 3
    K = 5

    C =  LaplacianKmodes(N, D, K, 1, 1, 1, mode='blobs')
    C.plot_3d_clusters(C=C.C)
    Z =  torch.ones(N,K) /K
    print(C.eval(Z))
