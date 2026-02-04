# Trajectory Anomaly Detection

Kernel-based anomaly detection methods for trajectory data, with applications to failure prediction in robotic systems.

## Installation

```bash
conda env create -f environment.yml
conda activate cd_project
```

## Quick Start

### Using the FFT-CD Detector

```python
import logging
import numpy as np
from detectors import get_method

# Enable diagnostic logging
logging.basicConfig(level=logging.INFO)

# Create the FFT-based kernel detector
detector = get_method("fft")

# Load your trajectory data: (n_episodes, episode_length, obs_dim)
# For example, 100 episodes of 200 timesteps with 3 features:
X_train = np.random.randn(100, 200, 3)

# Fit the detector
detector.fit(X_train)

# Predict on new windows (must be at least detector.window length)
X_test = np.random.randn(50, detector.window, 3)
predictions = detector.predict(X_test)  # 0 = normal, 1 = anomaly
scores = detector.score_samples(X_test)  # higher = more anomalous
```

Example output:
```
INFO:cd.detectors: Train/cal split: 70/30 episodes
INFO:cd.detectors.kernel: Window: 64 (auto-estimated, 1 period(s))
INFO:cd.algs.kern_cd: λ=1.00e-06 (adaptive), cond=4.52e+02, m=200
INFO:cd.detectors.kernel: Kernel: fft, γ=0.034
INFO:cd.detectors: Threshold: 5.67 (scores: 0.12/2.3/8.9 min/med/max, q=0.95)
```

### Logging Levels

Control verbosity via Python's logging module:

```python
import logging

# INFO: Key milestones (window, kernel params, threshold)
logging.basicConfig(level=logging.INFO)

# DEBUG: Additional details (window shapes, frequency estimation)
logging.basicConfig(level=logging.DEBUG)

# WARNING: Suppress diagnostic output
logging.getLogger("cd").setLevel(logging.WARNING)
```

## Failure Prediction Experiments

Run cross-validated failure prediction on the Upkie robot environment:

```bash
cd src
python scripts/fail_pred_results.py --env upkie --methods fft
```

Run all methods on all environments:

```bash
python scripts/fail_pred_results.py --env all
```

### Available Methods

| Key | Name | Description |
|-----|------|-------------|
| `fft` | FFT-CD | FFT-based kernel with auto window estimation |
| `sig` | Sig-CD | Signature kernel for path data |
| `rec` | ConvAE | Convolutional autoencoder (reconstruction) |
| `lat` | Conv-CD | ConvAE with kernel detector in latent space |

### Available Environments

| Environment | Description |
|-------------|-------------|
| `upkie` | Upkie wheeled biped robot |
| `hopper` | MuJoCo Hopper |
| `half_cheetah` | MuJoCo HalfCheetah |
| `ant` | MuJoCo Ant |
| `humanoid` | MuJoCo Humanoid |
| `inv_pend` | Inverted Pendulum |

## Core Components

### Detectors (`src/detectors/`)

- **KernDetector**: Kernel-based detector using the kernelized Christoffel-Darboux polynomials
- **ConvAEDetector**: Convolutional autoencoder for reconstruction-based detection

### Algorithms (`src/algs/`)

- **KernCD**: Core kernelized Christoffel-Darboux implementation with online updates
- **Kernels**: RBF, GaussFFT (for trajectories), SigKernel (signature kernel)
