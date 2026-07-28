"""Throw-away: compare moment-matrix conditioning on Hopper safe trajectories
when using (theta, z) vs (theta, z, theta_dot, z_dot)."""
import numpy as np

from cd.algs.cd_poly import CDPolynomial

# Hopper-v5 obs layout: 0=z, 1=theta, 2-4=joint angles, 5=xdot, 6=zdot,
# 7=theta_dot, 8-10=joint angular velocities.
DATA = 'scripts/misc/data/hopper_safe_mass1.00_n500.npz'
T_START, T_END = 200, 800

X = np.load(DATA)['X'][:, T_START:T_END]   # (n_traj, T, 11)
print(f"loaded {X.shape}: n_traj={X.shape[0]}, T={X.shape[1]}")

theta     = X[..., 1].ravel()
z         = X[..., 0].ravel()
theta_dot = X[..., 7].ravel()
z_dot     = X[..., 6].ravel()

P_2d = np.stack([theta, z], axis=-1)
P_4d = np.stack([theta, z, theta_dot, z_dot], axis=-1)

print(f"\nrange checks (raw, unscaled):")
for name, col in [('theta', theta), ('z', z), ('theta_dot', theta_dot), ('z_dot', z_dot)]:
    print(f"  {name:10s}  min={col.min():+.3f}  max={col.max():+.3f}  std={col.std():.3f}")

for degree in (3, 4, 5, 6):
    print(f"\n--- degree {degree} ---")
    cd2 = CDPolynomial(P_2d, degree=degree, basis='cheb', method='qr')
    cd4 = CDPolynomial(P_4d, degree=degree, basis='cheb', method='qr')
    c2 = float(np.linalg.cond(cd2.M))
    c4 = float(np.linalg.cond(cd4.M))
    print(f"  (theta, z)              n_terms={cd2.n_terms:4d}  cond(M)={c2:.3e}")
    print(f"  (theta, z, dot, dot)    n_terms={cd4.n_terms:4d}  cond(M)={c4:.3e}")
