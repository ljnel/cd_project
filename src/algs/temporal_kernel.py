import numpy as np
from scipy.spatial.distance import cdist

def build_temporal_kernel(t: np.ndarray, kernel_type: str, length_scale: float = 0.1, period: float = None):
    """
    Builds a (T, T) kernel matrix for temporal smoothing.
    
    t: 1D array of time steps (shape: T,)
    kernel_type: 'rbf', 'matern52', or 'periodic'
    length_scale: Controls how much to smooth (larger = smoother)
    period: Only used for the periodic kernel
    """
    # Reshape t to a column vector (T, 1) for distance calculations
    t_col = t.reshape(-1, 1)
    
    # Calculate pairwise distances
    sq_dists = cdist(t_col, t_col, metric='sqeuclidean')
    abs_dists = cdist(t_col, t_col, metric='euclidean')
    
    if kernel_type == 'rbf':
        # RBF / Squared Exponential
        K_t = np.exp(-sq_dists / (2 * length_scale**2))
        
    elif kernel_type == 'matern52':
        # Matérn v=5/2
        scaled_dist = np.sqrt(5) * abs_dists / length_scale
        K_t = (1 + scaled_dist + (scaled_dist**2) / 3) * np.exp(-scaled_dist)
        
    elif kernel_type == 'periodic':
        if period is None:
            raise ValueError("Must specify a 'period' for the periodic kernel.")
        # Periodic Kernel
        sine_term = np.sin(np.pi * abs_dists / period)
        K_t = np.exp(-2 * (sine_term / length_scale)**2)
        
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
        
    return K_t