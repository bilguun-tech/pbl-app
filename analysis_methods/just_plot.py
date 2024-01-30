import matplotlib.pyplot as plt


def just_plot(data, column_name):
    time = data.iloc[:, 0]
    value = data[column_name]
    fig, ax = plt.subplots()
    ax.plot(time, value)
    ax.set_title("Just Plot")
    return fig
