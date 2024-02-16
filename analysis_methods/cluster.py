# X-means法（エルボー法＝分散の減少率が急激に小さくなる点をクラスタ数に設定）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from kneed import KneeLocator


def cluster(original_df):
    df = original_df.copy()
    xlabel_name = df.columns[0]
    if xlabel_name == "年":  # 年ごとのデータの場合
        # df["index"] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01
        df[xlabel_name] = df.iloc[:, 0]
        
    elif xlabel_name == "date":  # 1日ごとの場合
        df[xlabel_name] = pd.to_datetime(df[xlabel_name])

    df.set_index(xlabel_name, inplace=True)
    # # 欠損値を処理（均値で置換）
    data = df.fillna(df.mean())

    # 商品名を行、日付を列に持つデータセットに変換
    grouped = data.T

    # tslearn用のデータセットに変換
    time_np = to_time_series_dataset(grouped)

    # クラスタ数の範囲を指定
    min_clusters = 2
    max_clusters = 10

    # エルボー法でのクラスタ数選択
    inertia_values = []

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(grouped)
        inertia_values.append(kmeans.inertia_)

    # KneeLocatorを使用してエルボーの位置を自動で選択
    kl = KneeLocator(
        range(min_clusters, max_clusters + 1),
        inertia_values,
        curve="convex",
        direction="decreasing",
    )
    best_n_clusters = kl.elbow  # 最適クラスタ数

    # 最適なクラスタ数を使用してX-means法を適用
    xmeans = TimeSeriesKMeans(
        n_clusters=best_n_clusters,
        metric="euclidean",
        verbose=True,
        random_state=0,
        n_init=1,
        max_iter=100,
    )
    labels_xmeans = xmeans.fit_predict(grouped)
    # print(labels_xmeans)

    # クラスタリングの結果
    # グラフを描画
    cmap = plt.get_cmap("tab10")  # 色の指定
    fig, ax = plt.subplots(figsize=(10, 8))

    # クラスタ中心のデータを保存するDataFrame
    cluster_centers_df = pd.DataFrame(index=grouped.T.index)
    # DataFrameに xlabel_name をインデックスとして設定
    cluster_centers_df[xlabel_name] = grouped.T.index
    cluster_centers_df.set_index(xlabel_name, inplace=True)
    for i in range(best_n_clusters):
        cluster_indices = np.where(labels_xmeans == i)[0]
        # クラスタ中心のデータをDataFrameに追加
        cluster_centers_df[f"Cluster{i} "] = xmeans.cluster_centers_[i].ravel()
        # クラスタ中心のみ描画（色を変更）
        ax.plot(
            grouped.T.index,
            xmeans.cluster_centers_[i].ravel(),
            label=f"クラスタ {i}",
            linewidth=2,
            color=cmap(i),
        )
        for idx in cluster_indices:
            # 中心以外のデータを透明に
            ax.plot(
                grouped.T.index,
                grouped.T.iloc[:, idx],
                label=f"{grouped.T.columns[idx]} ({i})",
                color=cmap(i),
                alpha=0,
            )

    # 列名に含まれている空白を削除する
    cluster_centers_df.columns = cluster_centers_df.columns.str.replace(" ", "")

    fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整
    ax.set_title(f"Clustering (Number of Cluster: {best_n_clusters})")
    ax.set_xlabel(xlabel_name, loc="right")
    ax.set_ylabel("Value", loc="top")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        prop={"size": 7},
        ncol=10,
    )

    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(7))
    # ax.xaxis.set_ticks([])
    plt.tight_layout()
    cluster_centers_df.reset_index(inplace=True)  # Reset the index
    # print(cluster_centers_df)

    return cluster_centers_df, fig
