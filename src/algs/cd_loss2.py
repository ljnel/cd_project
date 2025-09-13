import torch
from torch import nn
from algs.bases import *

class CDLoss(nn.Module):
    "CD polynomial implemented in pytorch so that it can be differentiated and updated in batches."

    def __init__(self, degree, n_vars, eps=1e-3, basis='mon'):
        """
        n_data: number of data points used to compute the moment matrix
        """
        super().__init__()
        self.n_vars = n_vars
        self.eps = eps

        bs = BasisSpec(n_vars=self.n_vars, degree=degree)
        assert basis in ["mon", "cheb"]
        if basis == "mon":
            self.basis = MonomialBasis(bs)
        if basis == "cheb":
            self.basis = ChebyshevBasis(bs)

        self.register_buffer("M_buf", torch.zeros(self.basis.n_terms, self.basis.n_terms))
        self.register_buffer("L_buf", torch.zeros(self.basis.n_terms, self.basis.n_terms))

        self.count = 0

    @torch.no_grad()
    def update_buffer(self, dl, ae, device):
        "Update the moment matrix over an entire epoch."
        M = torch.zeros(self.basis.n_terms, self.basis.n_terms).to(device)
        for x in dl:
            x = x.detach().to(device)
            V = self.basis.transform(ae.encode(x))
            M += (V.T @ V)
        M /= len(dl.dataset)
        I = torch.eye(V.shape[1]).to(M.device)
        M += self.eps * I
        L = torch.linalg.cholesky(M)

        self.M_buf.copy_(M)
        self.L_buf.copy_(L)

    def forward(self, x):

        v = self.basis.transform(x)
        
        y = torch.linalg.solve_triangular(self.L_buf, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        assert p_vals.shape == (x.shape[0],)

        return p_vals
    
    @torch.no_grad()
    def predict(self, x):
        assert not self.training
        v = self.basis.transform(x)
        y = torch.linalg.solve_triangular(self.L_buf, v.T, upper=False).T
        p_vals =  torch.einsum('bi,bi->b', y, y)
        assert p_vals.shape == (x.shape[0],)

        return p_vals