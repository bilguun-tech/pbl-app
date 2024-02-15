# 加法モデル

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def additive_method(df, column_name):
    xlabel_name = df.columns[0]
    if xlabel_name == "date":  # 1日ごとの場合
        df["index"] = pd.to_datetime(df['date'])
        num = 30 # 30日周期 
    else:
        df["index"] = pd.DataFrame(df.iloc[:, 0])
        # df["index"] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01
        num = 1 # 1周期

    df.set_index("index", inplace=True)
    data = df[column_name].dropna()  # NaNの行を削除

    # 成分分解
    result = seasonal_decompose(data, model="additive", period=num) #加法モデル
    # result=seasonal_decompose(df, model='multiplicative', period=num) #乗法モデル

    # グラフのサイズを設定
    plt.figure(figsize=(10, 8))

    # オリジナルデータのプロット
    plt.subplot(4, 1, 1)
    plt.plot(data, label="Original")
    plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.legend()

    # Trendデータのプロット
    plt.subplot(4, 1, 2, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.trend, label="Trend")
    plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.legend()

    # Seasonalデータのプロット
    plt.subplot(4, 1, 3, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.seasonal, label="Seasonal")
    plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.legend()

    # Residualデータのプロット
    plt.subplot(4, 1, 4, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.resid, label="Residual")
    plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.legend()

    # 横軸（時間軸）の目盛りを1ヶ月ごとに設定
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
    
    # グラフを表示
    plt.tight_layout()
    fig = plt.gcf()  # 現在のFigureを取得
    return fig
