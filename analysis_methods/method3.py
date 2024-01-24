import numpy as np
import matplotlib.pyplot as plt


def method3(data):
    x = np.linspace(0, 10, 100)
    y = x**2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Method 3 Plot")
    return fig
