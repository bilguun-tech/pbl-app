import pandas as pd
import matplotlib.pyplot as plt

def just_plot(df, column_name):
    # 前処理
    x_label = df.columns[0]
    df.set_index(df.iloc[:, 0], inplace=True)
    data = df[column_name].dropna()  # NaNの行を削除

    if (data == 0).mean() >= 0.1:   # 一割以上0があるデータ
        # 値が1以上である行をフィルタリング
        filtered_data = data[data >= 1]

        # 範囲を特定
        start_index = filtered_data.index.min()
        end_index = filtered_data.index.max()

        # 前後の行を指定
        start_index = data.index[data.index.get_loc(start_index) - 2]
        end_index = data.index[data.index.get_loc(end_index) + 2]

        subset_data = data.loc[start_index:end_index]

        # 可視化 
        plt.figure(figsize=(10,9))

        plt.subplot(2,1,1)
        plt.plot(data)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xlabel(x_label, fontname="MS Gothic")
        plt.title("Just Plot")

        plt.subplot(2,1,2)
        plt.plot(subset_data)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.xlabel(x_label, fontname="MS Gothic")
        plt.title("Subset Plot")

    else:  
        fig, ax = plt.subplots()
        data.plot()
        plt.xlabel(x_label, fontname="MS Gothic")
        ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
        ax.set_title("Just Plot")
    
    plt.tight_layout()
    fig = plt.gcf()
    return fig