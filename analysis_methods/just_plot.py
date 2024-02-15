import matplotlib.pyplot as plt


def just_plot(original_df, column_name):
    # time = data.iloc[:, 0]
    # value = data[column_name]
    # 前処理
    data = original_df.copy()
    x_label = data.columns[0]
    # data.set_index(data.iloc[:, 0], inplace=True)
    data = data[column_name].dropna()  # NaNの行を削除

    # 可視化
    fig, ax = plt.subplots()
    # ax.plot(time, value)
    data.plot()
    plt.xlabel(x_label)
    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    ax.set_title("Just Plot")

    return fig
