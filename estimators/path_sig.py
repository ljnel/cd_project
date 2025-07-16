import iisignature
from cd_poly.cd_poly import CDPolynomial
import numpy as np


def to_sincos(trajs, idx):

    sin = np.sin(trajs[:, idx])
    cos = np.sin(trajs[:, idx])

    return np.concatenate((trajs[:, :idx],
                           sin[:, np.newaxis],
                           cos[:, np.newaxis],
                           trajs[:, (idx+1):]), axis=1)


class PS():
    def __init__(self, ps_order, cd_degree, dim_reduction=None):
        self.order = ps_order
        self.deg = cd_degree
        self.dim_reduction = dim_reduction

    def fit(self, trajs):
        "trajs: (batch, chan, length)"
        trajs = to_sincos(trajs, 0)[:, :-1]
        ps = iisignature.sig(trajs.transpose(0, 2, 1), self.order) # (batch, ps_length)

        if self.dim_reduction is not None:
            ps = self.dim_reduction.fit_transform(ps)

        self.p = CDPolynomial(ps, self.deg, eps=1e-8)
        self.mean = self.p(ps).mean()

    def predict(self, trajs):
        trajs = to_sincos(trajs, 0)[:, :-1]
        ps = iisignature.sig(trajs.transpose(0, 2, 1), self.order)

        if self.dim_reduction is not None:
            ps = self.dim_reduction.transform(ps)

        vals = self.p(ps)
        return vals > self.mean


if __name__ == "__main__":

    from scipy import stats
    from simulation.gen_pend import gen_trajs
    from sklearn.decomposition import PCA

    n_normal = 1000
    n_ood = 1000
    n_sim = 10

    param_dists = {
        'q': stats.uniform(-np.pi, 2 * np.pi),
        'v': stats.norm(loc=0, scale=1.),
        'b': stats.uniform(loc=0, scale=0.01)
    }
    param_dists_ood = param_dists.copy()

    bs = np.linspace(0., 3., 10)
    accuracy = np.zeros((n_sim, len(bs)))

    for i in range(n_sim):
        for b_idx, b in enumerate(bs):
            _, trajs = gen_trajs(n_normal, 5., 64, param_dists)
            model = PS(ps_order=5, cd_degree=3, dim_reduction=PCA(n_components=4))
            model.fit(trajs)

            param_dists_ood['b'] = stats.uniform(loc=b, scale=0)
            _, trajs_ood = gen_trajs(n_ood, 5., 64, param_dists_ood)
            pred = model.predict(trajs_ood)
            accuracy[i, b_idx] = pred.mean()

    print(accuracy.mean(axis=0))