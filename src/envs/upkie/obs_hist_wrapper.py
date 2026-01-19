import numpy as np
import gymnasium as gym

class ObsHistoryWrapper(gym.Wrapper):
    """
    Stacks observation history + previous actions for PPO. 
    
    Output Shape: (history_length, obs_dim + action_dim)
    """
    
    def __init__(self, env, history_length: int = 10, obs_dim: int = 4, action_dim: int = 1):
        super().__init__(env)
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self._history = None
        self._last_action = np.zeros(action_dim, dtype=np.float32)
        
        shape = (history_length, obs_dim + action_dim)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._last_action.fill(0)
        # Fill the last row with current observation
        self._history[-1, :self.obs_dim] = obs
        return self._history.copy(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Update last action
        self._last_action[:] = np.atleast_1d(action).flatten()
        
        # Roll history and update latest entry
        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = np.concatenate([obs, self._last_action])
        
        return self._history.copy(), reward, term, trunc, info
    
    @property
    def raw_obs(self):
        """Returns the most recent raw observation (without action history)."""
        return self._history[-1, :self.obs_dim].copy()