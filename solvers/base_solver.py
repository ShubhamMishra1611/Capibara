from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np

class BaseSolver(ABC):
    def __init__(self, 
                 df_dx: Callable[[float, float], float], 
                 h: float, 
                 x_range: Tuple[float, float] = (0, 10), 
                 init_cond: Tuple[float, float] = (0, 0), 
                 *args)->None:
        self.df_dx = df_dx
        self.h = h
        self.x_range = x_range
        self.init_cond = init_cond
        self.args = args

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
