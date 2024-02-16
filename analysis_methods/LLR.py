# 局所線形回帰分析:Local linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter


def LLR(original_df, column_name):
    df = original_df.copy()
    xlabel_name = df.columns[0]
    if xlabel_name == "年":  # 年ごとのデータの場合
        df[xlabel_name] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01

    elif xlabel_name == "date":  # 1日ごとの場合
        df[xlabel_name] = pd.to_datetime(df[xlabel_name])

    tau = 1.0  # カーネル幅の設定（大きいほど、広い範囲で回帰を行う）
    xlim_start = df.iloc[0, 0]
    xlim_end = df.iloc[-1, 0]

    # XとYの列を取得
    X = df[xlabel_name].values  # 日付データのまま取得
    y = df[column_name].values

    # 局所線形回帰の関数を定義
    def calculate_weights(x, x_i, tau):
        import warnings

        # 警告を無視する
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # x_i を datetime から float に変換
        x_i = x_i.timestamp() if isinstance(x_i, pd.Timestamp) else x_i

        # 重み付け関数の計算
        weights = np.exp(-((x - x_i) ** 2) / (2 * tau**2))
        return np.diag(weights)

    # 局所線形回帰
    def locally_weighted_regression(x, y, x_i, tau):
        x = x.astype(float)  # x を float 型に変換
        x_i = float(x_i)  # x_i を float 型に変換
        weights = calculate_weights(x, x_i, tau)
        model = LinearRegression()
        model.fit(X=x.reshape(-1, 1), y=y, sample_weight=weights.diagonal())
        return model.predict([[x_i]])

    # 結果の表示
    def show_loess_regression(X, y, y_pred, tau, xlim_start, xlim_end):
        # 新しい図を作成し、サイズを調整
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X, y, color="black", label="Data")
        ax.plot(X, y_pred, color="red", label="LOESS Regression")

        # x軸のフォーマットを設定
        # date_format = DateFormatter("%Y-%m-%d")
        # ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整

        plt.xlabel(xlabel_name, fontname="MS Gothic")
        plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
        plt.legend()
        # x軸の範囲を指定
        plt.xlim(xlim_start, xlim_end)
        # 図を表示
        return fig

    # tauを設定して局所線形回帰を実行
    y_pred = [locally_weighted_regression(X, y, x_i, tau) for x_i in X]

    # 結果を表示
    error_msg = "No Error"
    fig = show_loess_regression(X, y, y_pred, tau, xlim_start, xlim_end)
    return fig, error_msg

