import matplotlib.pyplot as plt


def method1(data):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Method 1 Plot")
    return fig
