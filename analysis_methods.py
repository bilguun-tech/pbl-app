import numpy as np
import matplotlib.pyplot as plt


def method1(data):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title("Method 1 Plot")
    return plt.gcf()


def method2(data):
    x = np.linspace(0, 10, 100)
    y = np.cos(x)
    plt.plot(x, y)
    plt.title("Method 2 Plot")
    return plt.gcf()


def method3(data):
    x = np.linspace(0, 10, 100)
    y = x**2
    plt.plot(x, y)
    plt.title("Method 3 Plot")
    return plt.gcf()
