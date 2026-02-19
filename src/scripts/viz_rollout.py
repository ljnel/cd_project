from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ep', type=int)
    args = parser.parse_args()

    npz = np.load('hopper/train.npz', mmap_mode='r')
    states, actions, params = npz['states'][args.ep], npz['actions'][args.ep], npz['params'][args.ep]
    damp_scale, mass_scale, fric_scale = params

    env = gym.make('Hopper-v5', render_mode='human')
    model = SAC.load('hopper/sac_hopper.zip', env=env)
    m = env.unwrapped.model
    m.dof_damping[:] *= damp_scale
    m.body_mass[:] *= mass_scale
    m.geom_friction[:] *= fric_scale

    obs, _ = env.reset()
    terminated = False
    print(env.damp)

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        next_obs, r, terminated, truncated, info = env.step(action)
        
        # overlay BEFORE rendering so it shows up this frame
        if hasattr(gym_env.unwrapped, "viewer") and gym_env.unwrapped.viewer is not None:
            gym_env.unwrapped.viewer.add_overlay(
                kind=1,  # upper-left
                text1=f"Episode {episode} Step {i}",
                text2=f"Param: {env.damp:.2f}"
            )

        # now render with overlay applied
        gym_env.render()

        # define a failure condition for half cheetah
        if cheetah and abs(obs[1]) > 1.0: # fail if too much tilt
            terminated = True

        obs = next_obs

    env.close()