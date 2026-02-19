import torch
from torch import nn
from algs.poly_basis import *

class CDLoss(nn.Module):
    "CD polynomial implemented in pytorch so that it can be differentiated and updated in batches."

    def __init__(self, degree, n_vars, beta=.1, eps=1e-3, basis='mon'):
        """
        n_data: number of data points used to compute the moment matrix
        """
        super().__init__()
        self.n_vars = n_vars
        self.beta = beta
        self.eps = eps

        bs = BasisSpec(n_vars=self.n_vars, degree=degree)
        assert basis in ["mon", "cheb"]
        if basis == "mon":
            self.basis = MonomialBasis(bs)
        if basis == "cheb":
            self.basis = ChebyshevBasis(bs)

        self.register_buffer("M_buf", torch.zeros(self.basis.n_terms, self.basis.n_terms))
        self.count = 0

    @torch.no_grad()
    def update_buffer(self, x):
        "Update the buffer based on new inliers."
        V = self.basis.transform(x.detach())
        M_batch = (V.T @ V) / len(x)
        I = torch.eye(V.shape[1], device=V.device)
        M = (1 - self.beta) * self.M_buf + self.beta * M_batch + self.eps * I
        self.M_buf.copy_(M)

    def forward(self, x_in, x_out):
        "Compute the CD poly loss based on new inlier and outlier data. Also updates the buffer."

        V = self.basis.transform(x_in)
        v = self.basis.transform(x_out)

        M_batch = (V.T @ V) / len(x_in)
        I = torch.eye(V.shape[1], device=V.device)
        if self.count == 0:
            M = M_batch + self.eps * I
        else:
            M = (1 - self.beta) * self.M_buf + self.beta * M_batch + self.eps * I

        L = torch.linalg.cholesky(M)
        y = torch.linalg.solve_triangular(L, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        assert p_vals.shape == (x_out.shape[0],)

        self.M_buf.copy_(M.detach())
        self.count += 1

        return p_vals
    
    @torch.no_grad()
    def predict(self, x):
        assert not self.training
        v = self.basis.transform(x)
        L = torch.linalg.cholesky(self.M_buf)
        y = torch.linalg.solve_triangular(L, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        assert p_vals.shape == (x.shape[0],)

        return p_vals