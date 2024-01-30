# 加法モデル
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


<<<<<<< HEAD
<<<<<<< HEAD
def additive_method(df, column_name):
    df["date"] = df.iloc[:, 0]
    df = df[[column_name, "date"]]
=======
def additive_method(data, column_name):
    # amazonの株価（全期間）を加法モデルで分解
    df = pd.DataFrame(data)
    df = df[["date", column_name]]
>>>>>>> 458c1f0 (Fix additive method)
=======
def additive_method(df, column_name):
    df["date"] = df.iloc[:, 0]
    df = df[[column_name, "date"]]
>>>>>>> 9e33400 (delete unnecessary datasets and improve arima and additive method source)
    df.set_index("date", inplace=True)
    # print(df.head())

    # 成分分解
    result = seasonal_decompose(df, model="additive", period=30)  # 30日周期 #加法モデル
    # result=seasonal_decompose(df, model='multiplicative', period=30) #乗法モデル
    # グラフ化
    # result.plot()
    # plt.show()

    # グラフのサイズを設定
    plt.figure(figsize=(10, 8))

    # オリジナルデータのプロット
    plt.subplot(4, 1, 1)
    plt.plot(df[column_name], label="Original")
    plt.legend()

    # Trendデータのプロット
    plt.subplot(4, 1, 2, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.trend, label="Trend")
    plt.legend()

    # Seasonalデータのプロット
    plt.subplot(4, 1, 3, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.seasonal, label="Seasonal")
    plt.legend()

    # Residualデータのプロット
    plt.subplot(4, 1, 4, sharex=plt.gca())  # sharexを使用して横軸を共有
    plt.plot(result.resid, label="Residual")
    plt.legend()

    # 横軸（時間軸）の目盛りを1ヶ月ごとに設定
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))

    # グラフを表示
    plt.tight_layout()
    # plt.show()
    # グラフを非表示にしてfigを返す
    # plt.close()  # グラフを閉じる
    fig = plt.gcf()  # 現在のFigureを取得
    return fig
