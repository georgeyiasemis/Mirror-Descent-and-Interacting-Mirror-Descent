import torch
import numpy as np
import scipy.sparse as sp

class LinearSystem():

    def __init__(self, m, d, condition=None, sigma=0):
        assert m >= d
        self.m = m
        self.d = d
        self.W = self.generate_W(condition=condition)
        self.W_cond = np.linalg.cond(self.W.numpy())
        print('Condition of W = ', self.W_cond)
        self.b = self.generate_b()
        self.stochastic = True if sigma > 0 else False
        self.sigma = sigma

    def eval(self, x):
        self.calculate_loss(x)
        self.calculate_gradient(x)

    def generate_W(self, condition):

        if condition == None:
            W = torch.randn(self.m, self.d)
        else:
            W = torch.randn(self.m, self.d)
            U, S, V = torch.svd(W)
            S[S != 0] = torch.linspace(condition, 1, min(self.m, self.d))
            W = U @ torch.diag(S) @ V.T
        # print(W)
        return torch.real(W)

    def generate_b(self):

        return torch.randn(self.m)

    def calculate_loss(self, x):
        x = x.squeeze()
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        t = self.W @ x.squeeze() - self.b
        self.loss = (torch.norm(t) ** 2).item()
        if self.stochastic:
            self.loss += self.sigma * torch.randn(1).item()

    def calculate_gradient(self, x):
        x = x.squeeze()
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        t = self.W @ x - self.b
        self.gradient = 2 * self.W.T @ t
        if self.stochastic:
            self.gradient += self.sigma * torch.randn(self.d)
if __name__ == '__main__':
    m,  d = 100, 100
    condition = None
    sigma = 0.5
    # torch.manual_seed(20) 400 537
    for i in range(10000):
        torch.manual_seed(i)
        S = LinearSystem(m, d, condition, sigma)
        if S.W_cond <= 203 and S.W_cond >= 200:
            print(i)
            break
        # x = torch.randn(d)
        # S.calculate_loss(x)
        # S.calculate_gradient(x)
        # print(S.loss, S.gradient)
