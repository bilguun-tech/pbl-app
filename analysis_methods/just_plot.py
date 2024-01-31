import pandas as pd
import matplotlib.pyplot as plt

def just_plot(df, column_name):
    # 前処理
    x_label = df.columns[0]
    df.set_index(df.iloc[:, 0], inplace=True)
    data = df[column_name].dropna()  # NaNの行を削除

    print (data)
    if 0 in data.values:   # 0があるデータ
        # 値が1以上である行をフィルタリング
        filtered_data = data[data >= 1]

        # 範囲を特定
        start_index = filtered_data.index.min()
        end_index = filtered_data.index.max()

        # 前後の行を指定
        start_index = data.index[data.index.get_loc(start_index) - 1]
        end_index = data.index[data.index.get_loc(end_index) + 1]

        data = data.loc[start_index:end_index]

    # 可視化
    fig, ax = plt.subplots()
    data.plot()
    plt.xlabel(x_label, fontname="MS Gothic")
    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    ax.set_title("Just Plot")
    return fig