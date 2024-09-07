import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def plot_graphs(plots: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    plt.figure(figsize = (12, 6))
    for name, x, y in plots:
        plt.plot(x, y, 'o-', label = name)
    plt.title('ODE Solutions')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_amplitude_phase_errors():
    pass