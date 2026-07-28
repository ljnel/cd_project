"""Block-neural autoregressive flow density estimator — VectorDetector.

Score: -log p(x). Higher = more anomalous.
"""

import numpy as np


class FlowDetector:
    """Wrap a block-neural autoregressive flow (flowjax) as a VectorDetector."""

    def __init__(self, max_epochs: int = 20, lr: float = 5e-3, seed: int = 0):
        self.max_epochs = max_epochs
        self.lr = lr
        self.seed = seed

    def fit(self, x: np.ndarray) -> "FlowDetector":
        import jax.numpy as jnp
        import jax.random as jr
        from flowjax.distributions import Normal
        from flowjax.flows import block_neural_autoregressive_flow
        from flowjax.train import fit_to_data

        D = x.shape[-1]
        f = block_neural_autoregressive_flow(
            key=jr.key(self.seed),
            base_dist=Normal(jnp.zeros(D)),
        )
        self.flow, _ = fit_to_data(
            key=jr.key(self.seed + 1), dist=f, data=x,
            learning_rate=self.lr, max_epochs=self.max_epochs,
        )
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(-self.flow.log_prob(x))
