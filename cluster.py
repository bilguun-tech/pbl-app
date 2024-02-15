import matplotlib.pyplot as plt
import pandas as pd


def cluster(csv_file):
#X-means法（エルボー法＝分散の減少率が急激に小さくなる点をクラスタ数に設定）

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tslearn.utils import to_time_series_dataset
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.cluster import KMeans
    from kneed import KneeLocator

    # データ読み込み（日付を解釈してインデックスに設定する）
    df = pd.read_csv(csv_file, encoding='utf-8', parse_dates=['date'], index_col='date')

    # 欠損値を処理（均値で置換）
    df = df.fillna(df.mean())
    # 商品名を行、日付を列に持つデータセットに変換
    grouped = df.T

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
    kl = KneeLocator(range(min_clusters, max_clusters + 1), inertia_values, curve="convex", direction="decreasing")
    best_n_clusters = kl.elbow #最適クラスタ数

    # 最適なクラスタ数を使用してX-means法を適用
    xmeans = TimeSeriesKMeans(n_clusters=best_n_clusters, metric='euclidean', verbose=True,random_state=0, n_init=1, max_iter=100)
    labels_xmeans = xmeans.fit_predict(grouped)
    #print(labels_xmeans)

# クラスタリングの結果
# グラフを描画
    cmap = plt.get_cmap('tab10')  # 色の指定
    fig, ax = plt.subplots(figsize=(10,8))

    # クラスタ中心のデータを保存するDataFrame
    cluster_centers_df = pd.DataFrame(index=grouped.T.index)
    # DataFrameに 'date' をインデックスとして設定
    cluster_centers_df['date'] = grouped.T.index
    cluster_centers_df.set_index('date', inplace=True)
    for i in range(best_n_clusters):
        cluster_indices = np.where(labels_xmeans == i)[0]
        # クラスタ中心のデータをDataFrameに追加
        cluster_centers_df[f'Cluster{i} '] = xmeans.cluster_centers_[i].ravel()

        # クラスタ中心のみ描画（色を変更）
        ax.plot(grouped.T.index, xmeans.cluster_centers_[i].ravel(), label=f'クラスタ {i} 中心', linewidth=2, color=cmap(i))
        for idx in cluster_indices:
            # 中心以外のデータを透明に
            ax.plot(grouped.T.index, grouped.T.iloc[:, idx], label=f'{grouped.T.columns[idx]} ({i})', color=cmap(i), alpha=0)

    # 列名に含まれている空白を削除する
    cluster_centers_df.columns = cluster_centers_df.columns.str.replace(' ', '')

    # DataFrameをCSVファイルに保存
    #cluster_centers_df.to_csv('cluster_centers_data.csv')

    fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整
    ax.set_title(f'Clustering (Number of Cluster: {best_n_clusters})')
    ax.set_xlabel('Date', loc='right')
    ax.set_ylabel('Value', loc='top')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), prop={'size': 7, 'family': 'MS Gothic'}, ncol=10)
    ax.axes.xaxis.set_ticks([])
    plt.tight_layout()
    
    return cluster_centers_df, fig