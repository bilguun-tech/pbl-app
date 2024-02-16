# IsorationForestモデルによる異常検知:Anomaly Detection

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def anomaly_detection(original_df, column_name):
    df = original_df.copy()
    xlabel_name = df.columns[0]
    if xlabel_name == "date":  # 1日ごとの場合
        df[xlabel_name] = pd.to_datetime(df[xlabel_name])
        xlim_start = df.iloc[0,0]
        # # 日付データをエポック秒に変換（RCFを使う時には整数値でないといけない）
        df[xlabel_name] = (df[xlabel_name] - pd.Timestamp(xlim_start)) // pd.Timedelta('1s')

    else:
        df[xlabel_name] = df.iloc[:, 0]

    # # 欠損値を処理（均値で置換）
    data = df.fillna(df.mean())

    # ランダムカットフォレストモデル(RCF)の構築
    RCF_model = IsolationForest()
    RCF_model.fit(data)
    # データポイントの異常スコアを予測
    scores = RCF_model.decision_function(data)

    # 異常スコアの分布を可視化
    plt.figure(figsize=(10, 8))
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

    if xlabel_name == "date":
        data[xlabel_name] = pd.Timestamp(xlim_start) + pd.to_timedelta(data[xlabel_name], unit='s')

    # 閾値を超えるデータポイントを異常として可視化
    anomalies = data[scores < threshold]
    plt.subplot(2, 1, 2)
    plt.scatter(data[xlabel_name], data[column_name], c='blue', label='Normal Data')
    plt.scatter(anomalies[xlabel_name], anomalies[column_name], c='red', label='Anomalies')
    plt.xlabel(xlabel_name, fontname="MS Gothic")
    plt.legend(prop={"family": "MS Gothic"})
    plt.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換

    plt.tight_layout()
    fig = plt.gcf()  # 現在のFigureを取得

    return fig