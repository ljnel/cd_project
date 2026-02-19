# tests for cd_loss.py (including consistency with cd_poly.py)

import torch

from algs.cd_loss import CDLoss


def test():
    cd_loss = CDLoss(degree=3, bufsize=1000, n_vars=10, eps=1e-6)
    assert not cd_loss.is_ready()
    x1 = torch.rand((600, 10))
    cd_loss.update_buffer(x1)
    assert not cd_loss.is_ready()

    x2 = torch.rand((600, 10))
    cd_loss.update_buffer(x2)
    assert cd_loss.is_ready()

    x_batch = torch.rand((100, 10), requires_grad=True)
    x_in, x_out = x_batch[:50], x_batch[50:]
    loss = -torch.log(cd_loss(x_in, x_out)).mean()
    loss.backward()
    assert x_batch.grad.shape == x_batch.shape
    