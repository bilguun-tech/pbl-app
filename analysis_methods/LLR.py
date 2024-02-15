# 局所線形回帰分析:Local linear regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter

def LLR(df, column_name):
    if "年" in df.columns:  # 年ごとのデータの場合
        xlabel_name = "年"
        df["index"] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01
        # df[column_name].dropna()  # NaNの行を削除

    elif "date" in df.columns:  # 1日ごとの場合
        xlabel_name = "date"
        df["index"] = pd.to_datetime(df['date'])
        df.set_index("index", inplace=True)
        # df[column_name].dropna()  # NaNの行を削除
    
    tau = 1.0 # カーネル幅の設定（大きいほど、広い範囲で回帰を行う）

    xlim_start = df.iloc[0,0]
    xlim_end = df.iloc[0,-1]
    print(df[xlabel_name])

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
        weights = np.exp(-(x - x_i)**2 / (2 * tau**2))
        return np.diag(weights)

    def locally_weighted_regression(x, y, x_i, tau):
        x = x.astype(float)  # x を float 型に変換
        x_i = float(x_i)  # x_i を float 型に変換
        weights = calculate_weights(x, x_i, tau)
        model = LinearRegression()
        model.fit(X=x.reshape(-1, 1), y=y, sample_weight=weights.diagonal())
        return model.predict([[x_i]])

    def show_loess_regression(X, y, y_pred, tau, xlim_start, xlim_end):
        # 新しい図を作成し、サイズを調整
        fig, ax = plt.subplots(figsize=(10,8))

        ax.scatter(X, y, color='black', label='Data')
        ax.plot(X, y_pred, color='red', label='LOESS Regression')

        # x軸のフォーマットを設定
        date_format = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整

        plt.xlabel(xlabel_name)
        plt.ylabel(column_name)
        plt.legend()
        # x軸の範囲を指定
        xlim_start = '2013-01-01'
        xlim_end = '2013-05-01'
        plt.xlim(pd.Timestamp(xlim_start), pd.Timestamp(xlim_end))
        # 図を表示
        #plt.show()
        return fig

    # tauを設定して局所線形回帰を実行
    y_pred = [locally_weighted_regression(X, y, x_i, tau) for x_i in X]

    # 結果を表示
    return show_loess_regression(X, y, y_pred, tau, xlim_start, xlim_end)