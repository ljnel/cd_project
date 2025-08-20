import torch
from torch import nn
from algs.bases import *

class CDLoss(nn.Module):
    "CD polynomial implemented in pytorch so that it can be differentiated and updated in batches."

    def __init__(self, degree, bufsize, n_vars, eps, basis='mon'):
        """
        n_data: number of data points used to compute the moment matrix
        """
        super().__init__()
        self.bufsize = bufsize
        self.n_vars = n_vars
        self.eps = eps
        self.register_buffer("buf", torch.empty(0, n_vars))

        bs = BasisSpec(n_vars=self.n_vars, degree=degree)
        assert basis in ["mon", "cheb"]
        if basis == "mon":
            self.basis = MonomialBasis(bs)
        if basis == "cheb":
            self.basis = ChebyshevBasis(bs)

    def is_ready(self):
        return len(self.buf) == self.bufsize

    @torch.no_grad()
    def update_buffer(self, x):
        "Update the buffer based on new inliers. Inliers added to the buffer will not compute to the gradients."
        x_detached = x.detach()
        if self.buf.numel() == 0:
            self.buf = x_detached[-self.bufsize:]
        else:
            self.buf = torch.cat([self.buf, x_detached], dim=0)[-self.bufsize:]

    def forward(self, x_in, x_out):
        "Compute the CD poly loss based on new inlier and outlier data."
        assert self.is_ready()

        X = torch.cat([self.buf, x_in])  # Does this make sense?
        V = self.basis.transform(X)

        v = self.basis.transform(x_out)

        I = torch.eye(V.shape[1], device=V.device)
        M = (V.T @ V) / (self.bufsize + len(x_in)) + self.eps * I # moving average
        L = torch.linalg.cholesky(M)
        y = torch.linalg.solve_triangular(L, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        assert p_vals.shape == (x_out.shape[0],)
        return p_vals
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(self.buf[:0], x)