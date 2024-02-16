import pandas as pd
import matplotlib.pyplot as plt

def just_plot(df, column_name):
    # 前処理
    print(df)
    x_label = df.columns[0]
    df["index"] = pd.DataFrame(df.iloc[:, 0])
    df.set_index(df["index"], inplace=True)
    data = df[column_name].dropna()  # NaNの行を削除
    print(data)
    
    if not (data == 0).all() and (data == 0).mean() >= 0.1:  # データが全て0でない、かつ一割以上0があるデータ
    # len(data) >= 1000 # データ数が1000以上
        # 値が1以上である行をフィルタリング
        filtered_data = data[data >= 1]

        # 範囲を特定
        start_index = filtered_data.index.min()
        end_index = filtered_data.index.max()

        # 2つ前後の行を指定（範囲が最初や最後から2つ以内の場合は、その行自体を指定）
        if start_index <= df.index[1]:
            start_index = df.index[0]
        else:
            start_index = df.index[df.index.get_loc(start_index) - 1]

        if end_index >= df.index[-2]:
            end_index = df.index[-1]
        else:
            end_index = df.index[df.index.get_loc(end_index) + 1]

        subset_data = data.loc[start_index:end_index] 
        reset_data = subset_data.reset_index(drop=True)  
        reset_filtered_data = reset_data[reset_data >= 1]

        # 可視化 
        plt.figure(figsize=(10, 8))

        plt.subplot(2,1,1)
        plt.plot(data)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.xlabel(x_label)
        plt.title("Just Plot")

        plt.subplot(2,1,2)
        subset_data.plot(kind='bar', width=0.5)  # 棒グラフで表示
        plt.xticks(reset_filtered_data.index, filtered_data.index.tolist())  # X軸の目盛りを設定
        plt.xlabel(x_label)
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