"""Throwaway: ensemble disagreement of MLPs trained on unit-square boundary."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.paths import get_output_dir

torch.set_default_dtype(torch.float32)

N_BOUNDARY = 400
K_VALUES = [5, 10]
HIDDEN_VALUES = [50, 100, 200]
EPOCHS = 2000
LR = 1e-2
GRID_N = 80
GRID_LIM = 1.5


def sample_square_boundary(n, rng):
    t = rng.uniform(0.0, 4.0, size=n)
    pts = np.empty((n, 2))
    side = np.floor(t).astype(int)
    u = t - side
    s = 2 * u - 1
    pts[side == 0] = np.stack([s[side == 0], -np.ones(np.sum(side == 0))], axis=1)
    pts[side == 1] = np.stack([np.ones(np.sum(side == 1)), s[side == 1]], axis=1)
    pts[side == 2] = np.stack([-s[side == 2], np.ones(np.sum(side == 2))], axis=1)
    pts[side == 3] = np.stack([-np.ones(np.sum(side == 3)), -s[side == 3]], axis=1)
    return pts


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(X, y, seed, hidden):
    torch.manual_seed(seed)
    model = MLP(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    return model


def variance_on_grid(X, y, K, hidden, grid_tensor, grid_shape):
    models = [train_one(X, y, seed=k, hidden=hidden) for k in range(K)]
    with torch.no_grad():
        preds = np.stack([m(grid_tensor).numpy() for m in models], axis=0)
    return preds.var(axis=0).reshape(grid_shape)


def main():
    rng = np.random.default_rng(0)
    X_np = sample_square_boundary(N_BOUNDARY, rng).astype(np.float32)
    y_np = np.zeros(N_BOUNDARY, dtype=np.float32)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    g = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
    GX, GY = np.meshgrid(g, g)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)
    G = torch.from_numpy(grid)

    nrows = len(K_VALUES)
    ncols = len(HIDDEN_VALUES)
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for i, K in enumerate(K_VALUES):
        for j, hidden in enumerate(HIDDEN_VALUES):
            print(f"training K={K}, hidden={hidden}...")
            var = variance_on_grid(X, y, K, hidden, G, GX.shape)
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1, projection="3d")
            ax.plot_surface(GX, GY, var, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.scatter(X_np[:, 0], X_np[:, 1], np.zeros(N_BOUNDARY), s=2, c="red")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("var")
            ax.set_title(f"K={K}, hidden={hidden}")

    fig.suptitle("Ensemble disagreement of MLPs on unit-square boundary", fontsize=14)
    plt.tight_layout()
    out = get_output_dir() / "nn_ensemble_disagreement.png"
    plt.savefig(out, dpi=140)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
