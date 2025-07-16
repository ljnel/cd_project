import numpy as np
from scipy import stats

from simulation.gen_pend import gen_trajs, gen_one_traj
from cd_poly.cd_poly import CDPolynomial

TIME = 5.           # simulation time
N_UNDAMPED = 1000   # num of undamped trajs
N_DAMPED = 10       # num of damped trajs per damping const
FRAMES = 128        # num of frames over which simulation takes place





def traj2ps(trajs):
    """
    Get the path signatures of a batch of trajs, using sincos representation and discarding the velocities.

    Input: trajs of shape (n_trajs, 2, length)
    Output: path signatures of shape (n_trajs, n_ps)
    """



if __name__ == "__main__":

    param_dists = {
        'q': stats.uniform(-np.pi, 2 * np.pi),
        'v': stats.norm(loc=0, scale=1.),
        'b': stats.uniform(loc=0, scale=0.01)
    }

    times, trajs = gen_trajs(1000, TIME, FRAMES, param_dists)
    trajs = to_sincos(trajs, 0)
    sigs = iisignature.sig(trajs)

    p = CDPolynomial(sigs, degree=2)

    damps = np.linspace(0, 1., 10)

    for b in damps:
        param_dists['b'] = b
        trajs_ood = gen_trajs(N_DAMPED, TIME, FRAMES, param_dists)



