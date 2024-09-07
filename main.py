import numpy as np
from solvers.euler import EulerSolver
from solvers.runge_kutta import RungeKuttaSolver
from utils import plot_graphs

def df_dx(x, y, k=-1, c=0):
    return k * y + c

def exact_solution(x_values, k=-1, c=0, y0=1):
    """
    Computes the exact solution for the ODE dy/dx = k * y + c
    when k = -1 and c = 0.

    For k != 0, the solution is y(x) = y0 * exp(k * x).
    """
    return y0 * np.exp(k * x_values)

def main():
    h = 0.1
    x_range = (0, 5)
    init_cond = (0, 1)
    
    # Solvers
    euler_solver = EulerSolver(df_dx, h, x_range, init_cond)
    rk_solver = RungeKuttaSolver(df_dx, h, x_range, init_cond)
    
    x_values_explicit, y_values_explicit = euler_solver.solve_explicit()
    x_values_implicit, y_values_implicit = euler_solver.solve_implicit()
    x_values_rk2, y_values_rk2 = rk_solver.solve_rk2()
    x_values_rk4, y_values_rk4 = rk_solver.solve_rk4()

    x_exact = np.linspace(*x_range, 100)
    y_exact = exact_solution(x_exact)

    solutions = [
        ("Euler Explicit", x_values_explicit, y_values_explicit),
        ("Euler Implicit", x_values_implicit, y_values_implicit),
        ("RK2", x_values_rk2, y_values_rk2),
        ("RK4", x_values_rk4, y_values_rk4),
        ("Exact Solution", x_exact, y_exact)
    ]
    plot_graphs(solutions)

if __name__ == "__main__":
    main()
