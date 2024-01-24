import numpy as np
import matplotlib.pyplot as plt


def method2(data):
    x = np.linspace(0, 10, 100)
    y = np.cos(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Method 2 Plot")
    return fig
