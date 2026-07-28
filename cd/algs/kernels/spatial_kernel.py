import numpy as np


def _median_gamma(dists_sq: np.ndarray) -> float:
    """Median heuristic: gamma = 1 / median(pairwise squared distances)."""
    med = np.median(dists_sq[dists_sq > 0])
    return 1.0 / med if med > 0 else 1.0


def fit_gammas(x: np.ndarray, n_samples: int = 1000) -> dict:
    """Compute per-subkernel gammas via the median heuristic.

    Args:
        x: (N, 45) or (N, T, 45) — if 3D, flattened over time.
        n_samples: subsample size for pairwise distances.

    Returns:
        dict with keys gamma_r, gamma_theta, gamma_v (gamma_z is hardcoded).
    """
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    assert x.shape[-1] == 45

    # Subsample for efficiency
    if len(x) > n_samples:
        idx = np.random.choice(len(x), n_samples, replace=False)
        x = x[idx]

    q = x[:, 1:5]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    theta = x[:, 5:22]
    v = x[:, 22:45]

    # Quaternion: 1 - dot_q^2
    dot_q = (q[:, None, :] * q[None, :, :]).sum(axis=-1)
    gamma_r = _median_gamma((1.0 - dot_q ** 2).ravel())

    # Joint angles (periodic distance)
    dtheta = theta[:, None, :] - theta[None, :, :]
    gamma_theta = _median_gamma((np.sin(dtheta / 2) ** 2).sum(axis=-1).ravel())

    # Velocities
    dv = v[:, None, :] - v[None, :, :]
    gamma_v = _median_gamma((dv ** 2).sum(axis=-1).ravel())

    return dict(gamma_r=gamma_r, gamma_theta=gamma_theta, gamma_v=gamma_v)


def humanoid_composite_kernel(
    obs1: np.ndarray,
    obs2: np.ndarray,
    gamma_z: float = 0.1,
    gamma_r: float = 0.25,      # 1/D = 1/4
    gamma_theta: float = 1 / 17,
    gamma_v: float = 1 / 23,
    additive: bool = False,
) -> np.ndarray:
    """
    Composite kernel for the Gymnasium Humanoid-v5 observation space (dims 0-44).
    Inputs have shape (..., 45). Returns shape (...).

    By default uses the product kernel (k_height * k_rot * k_joints * k_vel).
    Set additive=True for the sum kernel (0.25 * (k_height + ...)).
    """
    assert obs1.shape[-1] == 45, f"Expected qpos+qvel obs (45 dims), got {obs1.shape[-1]}"
    assert obs2.shape[-1] == 45, f"Expected qpos+qvel obs (45 dims), got {obs2.shape[-1]}"

    z1, z2 = obs1[..., 0:1], obs2[..., 0:1]
    q1 = obs1[..., 1:5]
    q2 = obs2[..., 1:5]
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    theta1, theta2 = obs1[..., 5:22], obs2[..., 5:22]
    v1, v2 = obs1[..., 22:45], obs2[..., 22:45]

    # Height: RBF
    k_height = np.exp(-gamma_z * (z1 - z2) ** 2)

    # Quaternion: squared dot-product on S^3
    dot_q = (q1 * q2).sum(axis=-1, keepdims=True)
    k_rot = np.exp(-gamma_r * (1.0 - dot_q ** 2))

    # Joint angles: periodic RBF
    k_joints = np.exp(-gamma_theta * (np.sin((theta1 - theta2) / 2) ** 2).sum(axis=-1, keepdims=True))

    # Velocities: RBF
    k_vel = np.exp(-gamma_v * ((v1 - v2) ** 2).sum(axis=-1, keepdims=True))

    if additive:  # noqa: SIM108  (ternary would exceed line length / hurt readability)
        k_total = 0.25 * (k_height + k_rot + k_joints + k_vel)
    else:
        k_total = k_height * k_rot * k_joints * k_vel
    return k_total.squeeze(-1)



def humanoid_kernel_matrices(
    states: np.ndarray, states2: np.ndarray | None = None, **kwargs
) -> np.ndarray:
    """Kernel matrix at each timestep.

    (N, T, D) -> (T, N, N)  or  (N, T, D), (M, T, D) -> (T, N, M)
    """
    obs1 = states.transpose(1, 0, 2)   # (T, N, D)
    obs2 = states2.transpose(1, 0, 2) if states2 is not None else obs1  # (T, M, D)
    return humanoid_composite_kernel(obs1[:, :, None, :], obs2[:, None, :, :], **kwargs)


def fit_rbf_gamma(x: np.ndarray, n_samples: int = 1000) -> float:
    """Median heuristic for a flat RBF kernel.

    Args:
        x: (N, D) or (N, T, D) — if 3D, flattened over time.
    """
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    if len(x) > n_samples:
        idx = np.random.choice(len(x), n_samples, replace=False)
        x = x[idx]
    diff = x[:, None, :] - x[None, :, :]
    return _median_gamma((diff ** 2).sum(axis=-1).ravel())


def rbf_kernel_matrices(
    states: np.ndarray, states2: np.ndarray | None = None, gamma: float | None = None
) -> np.ndarray:
    """RBF kernel on raw obs at each timestep. (N,T,D),(M,T,D) -> (T,N,M)."""
    N, T, D = states.shape
    if states2 is None:
        states2 = states
    if gamma is None:
        gamma = 1.0 / D
    obs1 = states.transpose(1, 0, 2)  # (T, N, D)
    obs2 = states2.transpose(1, 0, 2)  # (T, M, D)
    diff = obs1[:, :, None, :] - obs2[:, None, :, :]  # (T, N, M, D)
    return np.exp(-gamma * (diff ** 2).sum(axis=-1))
