from abc import ABC, abstractmethod
from typing import Optional, Tuple
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
    def get_train_test(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        "Return x_tr, y_tr (optional), x_te"
        pass

    @abstractmethod
    def eval(self, y_pred: np.ndarray) -> float:
        pass