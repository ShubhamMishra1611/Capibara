import numpy as np
from .base_solver import BaseSolver
from typing import Tuple

class EulerSolver(BaseSolver):

    def solve(self, method: str = "explicit") -> Tuple[np.ndarray]:
        if method == "explicit": return self.solve_explicit()
        elif method == "implicit": return self.solve_implicit()
        else: 
            raise f"INVALID method type {method} given in {self.__class__.__name__}"

    def solve_explicit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves an ODE using the Euler Explicit (Forward Euler) method.
        """
        x_min, x_max = self.x_range
        total_steps = int((x_max - x_min) / self.h) + 1
        x_values = np.linspace(x_min, x_max, total_steps)
        y_values = np.zeros(total_steps)

        x0, y0 = self.init_cond
        y_values[0] = y0

        for i in range(1, total_steps):
            y_values[i] = y_values[i-1] + self.h * self.df_dx(x_values[i-1], y_values[i-1], *self.args)

        return x_values, y_values

    def solve_implicit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves an ODE using the Euler Implicit method with the Trapezoidal rule.
        """
        x_min, x_max = self.x_range
        total_steps = int((x_max - x_min) / self.h) + 1
        x_values = np.linspace(x_min, x_max, total_steps)
        y_values = np.zeros(total_steps)

        x0, y0 = self.init_cond
        y_values[0] = y0

        for i in range(1, total_steps):
            x_prev = x_values[i-1]
            y_prev = y_values[i-1]
            x_next = x_values[i]

            y_guess = y_prev + self.h * self.df_dx(x_prev, y_prev, *self.args)

            for _ in range(10):  # Iterate to converge
                y_guess = y_prev + (self.h / 2) * (self.df_dx(x_prev, y_prev, *self.args) + self.df_dx(x_next, y_guess, *self.args))

            y_values[i] = y_guess

        return x_values, y_values
