import matplotlib.pyplot as plt


def just_plot(data, column_name):
    x = data.iloc[:, 0]
    y = data[column_name]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Just Plot")
    return fig
