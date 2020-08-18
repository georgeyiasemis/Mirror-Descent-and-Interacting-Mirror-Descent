import torch
import numpy as np
import scipy.sparse as sp

class LinearSystem():

    def __init__(self, M, d,  condition=None, sigma=0):
        self.M = M
        self.d = d
        self.Ws = []
        self.bs = []
        self.W = self.generate_W(condition=condition)
        self.W_cond = np.linalg.cond(self.W.numpy())
        print('Condition of W = ', self.W_cond)
        self.b = self.generate_b()
        self.sigma = sigma
        for i in range(M):
            self.Ws.append(self.W + torch.randn(self.W.shape) * 0.05)
            self.bs.append(self.b + torch.randn(self.b.shape) * 0.05)

        self.stochastic = True if sigma > 0 else False


    def eval(self, x, batch_size):
        ind = np.random.choice(range(0,self.M), batch_size, replace=False)
        self.loss = 0.0
        self.gradient = torch.zeros((self.d)).float()

        for k in ind:
            k = k.item()
            self.loss += self.calculate_loss(x, self.Ws[k], self.bs[k]) / batch_size
            self.gradient += self.calculate_gradient(x, self.Ws[k], self.bs[k]) / batch_size

    def generate_W(self, condition=None):
        """
        Generates randomly an m x d matrix W. If condition is not None  the
            condition of W is approximately that.

        Parameters
        ----------
        condition : float or None
            If not None the condition of W is approximately equal to condition.

        Returns
        -------
        W : torch.Tensor
            Tensor of shape (m, d).

        """
        if condition == None:
            W = torch.randn(self.d, self.d)
        else:
            W = torch.randn(self.d, self.d)
            U, S, V = torch.svd(W)
            S[S != 0] = torch.linspace(condition, 1, min(self.d, self.d))
            W = U @ torch.diag(S) @ V.T

        return torch.real(W).float()

    def generate_b(self):

        return torch.randn(self.d).float()

    def calculate_loss(self, x, W, b):
        """Calculate loss f(x):
            f(x) = 1/M sum_{i=1}^M f_i(x) = 1/M sum_{i=1}^M ||Wx_i - b||_2^2
            where M = batch_size.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, d) or (d,).

        """
        x = x.squeeze()
        assert isinstance(x, torch.Tensor)

        if len(x.shape) == 1:
            assert x.size(0) == self.d
            t = W @ x.squeeze() - b
            loss = (torch.norm(t) ** 2).item()
            if self.stochastic:
                loss += self.sigma * torch.randn(1).item()

            return loss

        else:
            raise ValueError('Input should be a 1-D')


    def calculate_gradient(self, x, W, b):
        """Calculate gradient f(x):
            \ nabla f(x) = 1 / M  sum_{i=1}^M 2 * W.T * (W * x_i -b).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, d) or (d,).

        """

        x = x.squeeze()
        assert isinstance(x, torch.Tensor)

        if len(x.shape) == 1:
            assert x.size(0) == self.d
            t = W @ x - b
            gradient = 2 * W.T @ t
            if self.stochastic:
                gradient += self.sigma * torch.randn(self.d)
            return gradient

        else:
            raise ValueError('Input should be a 1-D.')



if __name__ == '__main__':
    m,  d = 100, 100
    M = 20
    condition = None
    sigma = 0
    # torch.manual_seed(20) 400 537
    # for i in range(10000):
    #     torch.manual_seed(i)
    #     S = LinearSystem(m, d, condition, sigma)
    #     if S.W_cond <= 203 and S.W_cond >= 200:
    #         print(i)
    #         break
    torch.manual_seed(0)
    S = LinearSystem(M, d, condition, sigma)
    x = torch.randn(d)
    S.eval(x)
    print(S.loss, S.gradient)
