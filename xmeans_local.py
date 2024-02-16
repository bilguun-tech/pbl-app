#X-means クラスタ重心をプロット

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from kneed import KneeLocator

csv_file = "datasets/store_sales.csv"
#csv_file = "datasets/S&P500 stock prices.csv"

# データ読み込み（日付を解釈してインデックスに設定する）
df = pd.read_csv(csv_file,encoding='utf-8', parse_dates=['date'], index_col='date')
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
xmeans = TimeSeriesKMeans(n_clusters=best_n_clusters, metric='euclidean', verbose=True,random_state=42, n_init=1, max_iter=100)
labels_xmeans = xmeans.fit_predict(grouped)
#print(labels_xmeans)

# クラスタリングの結果
# ラフを描画
cmap = plt.get_cmap('tab10')  # 色の指定
fig, axs = plt.subplots(figsize=(10,8))
index = 0
for value in grouped.T.columns:
    #axs.plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
    axs.plot(grouped.T.index, label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
    index += 1

# クラスタリング重心の可視化
for cluster_idx in range(best_n_clusters):
   axs.plot(xmeans.cluster_centers_[cluster_idx].ravel(), color=cmap(cluster_idx))

print(xmeans.cluster_centers_[0].ravel())


fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整
axs.set_title(f'Clustering (Number of Clusters: {best_n_clusters})')
axs.set_xlabel('date', loc= 'right')
axs.set_ylabel('value', loc= 'top')
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), prop={'size': 6, 'family':'MS Gothic'},fontsize="small", ncol=5)
axs.axes.xaxis.set_ticks([])
plt.tight_layout()
plt.show()
