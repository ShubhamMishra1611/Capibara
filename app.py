import numpy as np
import streamlit as st
from solvers.euler import EulerSolver
from solvers.runge_kutta import RungeKuttaSolver
from utils import plot_graphs
from utils import calculate_amplitude_phase_errors
import matplotlib.pyplot as plt


def df_dx(x, y, k=-1, c=0):
    return k * y + c

def exact_solution(x_values, k=-1, c=0, y0=1):
    return y0 * np.exp(k * x_values)


st.title("ODE Solver and Visualizer")
st.sidebar.header("Parameters")

h = st.sidebar.slider("Step size (h)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
x_start = st.sidebar.number_input("x start", value=0.0)
x_end = st.sidebar.number_input("x end", value=5.0)
y0 = st.sidebar.number_input("Initial condition y0", value=1.0)


x_range = (x_start, x_end)
init_cond = (x_start, y0)

euler_solver = EulerSolver(df_dx, h, x_range, init_cond)
rk_solver = RungeKuttaSolver(df_dx, h, x_range, init_cond)

x_values_explicit, y_values_explicit = euler_solver.solve_explicit()
x_values_implicit, y_values_implicit = euler_solver.solve_implicit()
x_values_rk2, y_values_rk2 = rk_solver.solve_rk2()
x_values_rk4, y_values_rk4 = rk_solver.solve_rk4()


x_exact = np.linspace(x_start, x_end, 100)
y_exact = exact_solution(x_exact, y0=y0)


solutions = [
    ("Euler Explicit", x_values_explicit, y_values_explicit),
    ("Euler Implicit", x_values_implicit, y_values_implicit),
    ("RK2", x_values_rk2, y_values_rk2),
    ("RK4", x_values_rk4, y_values_rk4),
    ("Exact Solution", x_exact, y_exact)
]


fig, ax = plt.subplots(figsize=(10, 6))
for method_name, x_vals, y_vals in solutions:
    ax.plot(x_vals, y_vals, label=method_name)
ax.set_title("ODE Solutions Comparison for dy/dx = -y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)


st.pyplot(fig)
