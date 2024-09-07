# Differential Equation Solvers

This repository provides implementations of some numerical methods for solving differential equations of the form:

```math

\frac{df}{dx} = \lambda x + c
```

The included methods are:

- **Euler’s Method**
- **Runge-Kutta Methods (RK)**
- **Crank-Nicolson Method**

## Usage

Each method is implemented as a Python function in its respective file. Here’s a brief example of how to use these methods:

### Euler’s Method

```python
from solvers.euler import EulerSolver

def df_dx(x, y, k=-1, c=0):
    return k * y + c
h = 0.1
x_range = (0, 5)
init_cond = (0, 1)

euler_solver = EulerSolver(df_dx, h, x_range, init_cond)

x_values_explicit, y_values_explicit = euler_solver.solve_explicit()
x_values_implicit, y_values_implicit = euler_solver.solve_implicit()


solutions = [
        ("Euler Explicit", x_values_explicit, y_values_explicit),
        ("Euler Implicit", x_values_implicit, y_values_implicit),
    ]

plot_graphs(solutions)

```

To run thoguh interactive streamlit app just 
```bash
#!bin/bash
streamlit run app.py
```
