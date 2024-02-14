#IsorationForestモデルによる異常検知

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def method2(csv_file):
    # 任意の特徴量を選択してx軸とy軸に割り当てる
    x_feature = 'date'
    y_feature = 'BEAUTY'

    # CSVファイルからデータを読み込む（日付を解釈する）
    df = pd.read_csv(csv_file, parse_dates=[x_feature])

    # 日付データをエポック秒に変換
    df[x_feature] = (df[x_feature] - pd.Timestamp("2013-01-01")) // pd.Timedelta('1s')

    # 欠損値を処理（均値で置換）
    df = df.fillna(df.mean())

    # ランダムカットフォレストモデルの構築
    model = IsolationForest()
    model.fit(df)
    # データポイントの異常スコアを予測
    scores = model.decision_function(df)

    plt.figure(figsize=(10,8))
    #異常スコアの分布を可視化
    plt.subplot(2, 1, 1)
    plt.hist(scores, bins=50, density=True, alpha=0.5, color='blue', label='Anomaly Scores')

    # 適切な閾値を設定（-1～0の範囲、閾値よりも小さい値が異常値と判断される）
    threshold = -0.1

    #閾値の範囲を表示するコード
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    plt.title('Distribution of Anomaly Scores with Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()

    # 閾値を超えるデータポイントを異常として可視化
    anomalies = df[scores < threshold]
    plt.subplot(2, 1, 2)
    plt.scatter(df[x_feature], df[y_feature], c='blue', label='Normal Data')
    plt.scatter(anomalies[x_feature], anomalies[y_feature], c='red', label='Anomalies')
    plt.ylabel(f'{y_feature}')
    plt.legend(prop={"family": "MS Gothic"})

    plt.tight_layout()
    fig = plt.gcf()  # 現在のFigureを取得

    return fig
