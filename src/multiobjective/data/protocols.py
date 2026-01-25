from typing import Protocol

import numpy as np


class DatasetLoader(Protocol):
    def load(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...
