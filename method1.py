#局所線形回帰分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter

'''変更前コード
def method1(csv_file):
    #csv_file = 'datasets/store_sales.csv'
    x_column = 'date'
    y_column = 'BEAUTY'
    tau = 1.0
    xlim_start = '2013-01-01'
    xlim_end = '2013-05-01'
    # CSVファイルからデータを読み込む（日付を解釈する）
    df = pd.read_csv(csv_file, parse_dates=[x_column])

    # XとYの列を取得
    X = df[x_column].values  # 日付データのまま取得
    y = df[y_column].values

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
        fig, ax = plt.subplots(figsize=(30, 18))

        ax.scatter(X, y, color='black', label='Data')
        ax.plot(X, y_pred, color='red', label='LOESS Regression')

        # x軸のフォーマットを設定
        date_format = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend()
        # x軸の範囲を指定
        xlim_start = '2013-01-01'
        xlim_end = '2013-05-01'
        plt.xlim(pd.Timestamp(xlim_start), pd.Timestamp(xlim_end))
        plt.tight_layout()
        #plt.show()
        # グラフを非表示にしてfigを返す
        #plt.close()  # グラフを閉じる
        fig = plt.gcf()  # 現在のFigureを取得
        return fig


    # tauを設定して局所線形回帰を実行
    y_pred = [locally_weighted_regression(X, y, x_i, tau) for x_i in X]

    # 結果を表示
    show_loess_regression(X, y, y_pred, tau, xlim_start, xlim_end)'''


def method1(csv_file):
    #csv_file = 'datasets/store_sales.csv'
    x_column = 'date'
    y_column = 'BEAUTY'
    xlim_start = '2013-01-01'
    xlim_end = '2013-05-01'
    #カーネル幅の設定（大きいほど、広い範囲で回帰を行う）
    tau = 1.0
    # CSVファイルからデータを読み込む（日付を解釈する）
    df = pd.read_csv(csv_file, parse_dates=[x_column])

    # XとYの列を取得
    X = df[x_column].values  # 日付データのまま取得
    y = df[y_column].values

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

        plt.xlabel(x_column)
        plt.ylabel(y_column)
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
