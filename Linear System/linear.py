import torch
import numpy as np
import scipy.sparse as sp

class LinearSystem():

    def __init__(self, m, d, condition=None, sigma=0):

        """This class contains methods to solve the optimisation problem
            of the linear system (Equation 5.3)
                \min f(x) = || W @ x - b ||_2^2.

        Parameters
        ----------
        m, n : int, int
            Dimensions of W.
        condition : float
            Condition of W.
        sigma : float
            Noise parameter. If sigma > 0 then optimisation is Stochastic.

        """
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

        """
        Parameters
        ----------
        x : torch.tensor
            Input vector of shape (d,) or (d,1).

        """
        self.calculate_loss(x)
        self.calculate_gradient(x)

    def generate_W(self, condition):

        """Generates randomly a matrix W with a specific condition number.

        Parameters
        ----------
        condition : float
            Condition number .

        Returns
        -------
        W : torch.tensor

        """


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

        """Calculates loss function value at x:

            f(x) = || W @ x - b ||_2^2

        Parameters
        ----------
        x : torch.tensor
            Input vector of shape (d,) or (d,1).

        """
        x = x.squeeze()
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1

        t = self.W @ x.squeeze() - self.b

        self.loss = (torch.norm(t) ** 2).item()

        if self.stochastic:
            self.loss += self.sigma * torch.randn(1).item()

    def calculate_gradient(self, x):

        """Calculates the gradient at x:

            \ nabla f(x) = 2 ( W @ x - b )

        Parameters
        ----------
        x : torch.tensor
            Input vector of shape (d,) or (d,1).

        """

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
