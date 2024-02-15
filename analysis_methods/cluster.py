import matplotlib.pyplot as plt
import pandas as pd


def cluster(data):
    # Do the clustering ...

    # Sample data after clustering
    sample_data = {
        "Year": [2020, 2021, 2022, 2023, 2024],
        "Cluster 1": [25, 30, 35, 40, 45],
        "Cluster 3": [50000, 60000, 70000, 80000, 90000],
    }
    df = pd.DataFrame(sample_data)

    x_label = df.columns[0]

    # Visualization
    fig, ax = plt.subplots()
    for column in df.columns[1:]:
        ax.plot(df[x_label], df[column], label=column)
    plt.xlabel(x_label, fontname="MS Gothic")
    ax.ticklabel_format(
        style="plain", axis="y"
    )  # Convert to normal notation from exponential notation
    ax.set_title("Cluster plot")
    ax.legend()

    return df, fig
