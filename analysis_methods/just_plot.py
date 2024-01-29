import matplotlib.pyplot as plt


def just_plot(data):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Just Plot")
    return fig
