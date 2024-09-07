import numpy as np
from .base_solver import BaseSolver
from typing import Tuple


class CrankNicolsonSolver(BaseSolver):
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        return None, None
