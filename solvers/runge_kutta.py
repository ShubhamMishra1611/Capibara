import numpy as np
from .base_solver import BaseSolver
from typing import Tuple

class RungeKuttaSolver(BaseSolver):

    def solve(self, method: str = "rk2") -> Tuple[np.ndarray]:
        if method == "rk2": return self.solve_rk2()
        elif method == "rk4": return self.solve_rk4()

        else: 
            raise f"INVALID method type {method} given in {self.__class__.__name__}"
    
    def solve_rk2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the ODE using Runge-Kutta 2nd order (RK2) method.
        """
        x_min, x_max = self.x_range
        total_steps = int((x_max - x_min) / self.h) + 1
        x_values = np.linspace(x_min, x_max, total_steps)
        y_values = np.zeros(total_steps)

        # Initial conditions
        x0, y0 = self.init_cond
        y_values[0] = y0

        for i in range(1, total_steps):
            k1 = self.h * self.df_dx(x_values[i-1], y_values[i-1], *self.args)
            k2 = self.h * self.df_dx(x_values[i-1]+ self.h, y_values[i-1] + k1, *self.args)
            
            y_values[i] = y_values[i-1] + 0.5 * (k1 + k2)
        
        return x_values, y_values
    
    def solve_rk4(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the ODE using Runge-Kutta 2nd order (RK2) method.
        """
        x_min, x_max = self.x_range
        total_steps = int((x_max - x_min) / self.h) + 1
        x_values = np.linspace(x_min, x_max, total_steps)
        y_values = np.zeros(total_steps)

        # Initial conditions
        x0, y0 = self.init_cond
        y_values[0] = y0

        for i in range(1, total_steps):
            k1 = self.h * self.df_dx(x_values[i-1], y_values[i-1], *self.args)
            k2 = self.h * self.df_dx(x_values[i-1] + 0.5 * self.h, y_values[i-1] + 0.5 * k1, *self.args)
            k3 = self.h * self.df_dx(x_values[i-1] + 0.5 * self.h, y_values[i-1] + 0.5 * k2, *self.args)
            k4 = self.h * self.df_dx(x_values[i-1] + self.h, y_values[i-1] + k3, *self.args)

            y_values[i] = y_values[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return x_values, y_values
