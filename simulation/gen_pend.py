import pinocchio as pin
import numpy as np


def create_model(mass, length):
    "Create a Pinocchio model for an inverted pendulum."
    model = pin.Model()

    joint_id = model.addJoint(0,
                              pin.JointModelRX(),
                              pin.SE3.Identity(),
                              "pendulum_joint"
                              )
    inertia = pin.Inertia(mass,
                          np.array([0., 0., length / 2]),
                          pin.Inertia.FromCylinder(mass,
                                                   length / 20,
                                                   length).inertia)
    model.appendBodyToJoint(joint_id,
                            inertia,
                            pin.SE3.Identity())
    model.gravity.linear[2] = -9.81

    return model


def simulate_inverted_pendulum(model, t, steps, params):
    init_q, init_v, damping_coeff = params  # angle, ang vel, damping coeff
    dt = t / steps

    data = model.createData()

    q = np.array([init_q])
    v = np.array([init_v])

    trajectory = []

    for i in range(steps):
        trajectory.append([q.copy(), v.copy()])

        tau = np.array([-damping_coeff * v])
        a = pin.aba(model, data, q, v, tau)

        v += a * dt
        q = pin.integrate(model, q, v * dt)

    return np.array(trajectory).T


def gen_trajs(n_simulations, t, steps, param_dists, mass=1.0, length=1.0):
    """
    Simulate multiple trajectories with given param distributions.
    
    param_dists: a dictionary containing, for each desired parameter, either
    a scipy.stats distribution or a number.
    """
    qs = param_dists['q'].rvs(n_simulations)
    vs = param_dists['v'].rvs(n_simulations)
    bs = param_dists['b'].rvs(n_simulations)

    times = np.linspace(0., t, steps)
    trajs = np.zeros((n_simulations, 2, steps))

    model = create_model(mass, length)

    for i in range(n_simulations):

        q = qs[i]
        v = vs[i]
        b = bs[i]
        params = (q, v, b)

        traj = simulate_inverted_pendulum(
            model, t, steps, params
        )
        trajs[i] = traj

    trajs = np.array(trajs).reshape(n_simulations, 2, -1)
    return times.astype(np.float32), trajs.astype(np.float32)


def gen_one_traj(t, steps, q, v, b):

    times = np.linspace(0, t, steps)
    model = create_model()

    q = np.array(q)
    v = np.array(v)
    b = np.array(b)
    params = (q, v, b)

    traj = simulate_inverted_pendulum(
        model, t, steps, params
    )

    return times.astype(np.float32), traj.astype(np.float32)
