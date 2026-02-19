from abc import ABC, abstractmethod

import numpy as np

"""
An abstract base class for objects which do the following:
    - set up an experiment from some params
    - query it to get the train/test data
    - submit your predictions to get a score
"""

class Experiment(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def get_train_test(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        "Return x_tr, y_tr (optional), x_te"
        pass

    @abstractmethod
    def eval(self, y_pred: np.ndarray) -> float:
        pass