import numpy as np
import matplotlib.pyplot as plt


def method3(csv_file):
#!/usr/bin/env python
# coding: utf-8

#X-means法（エルボー法＝分散の減少率が急激に小さくなる点をクラスタ数に設定）

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tslearn.utils import to_time_series_dataset
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.cluster import KMeans
    from kneed import KneeLocator

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
    xmeans = TimeSeriesKMeans(n_clusters=best_n_clusters, metric='euclidean', verbose=True,random_state=0, n_init=1, max_iter=100)
    labels_xmeans = xmeans.fit_predict(grouped)
    #print(labels_xmeans)

    # クラスタリングの結果
# グラフを描画
    cmap = plt.get_cmap('tab10')  # 色の指定
    fig, axs = plt.subplots(figsize=(10,8))
    index = 0
    for value in grouped.T.columns:
        axs.plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
        index += 1
    fig.autofmt_xdate()  # 日付が重ならないようにフォーマットを調整

    axs.set_title(f'Clustering (Number of Clusters: {best_n_clusters})')
    axs.set_xlabel('date', loc= 'right')
    axs.set_ylabel('value', loc= 'top')
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), prop={'size': 6, 'family':'MS Gothic'},fontsize="small", ncol=5)
    axs.axes.xaxis.set_ticks([])
    plt.tight_layout()
    return fig


    '''# グラフを描画（subplot(2,1,1)）
    index = 0
    for value in grouped.T.columns:
        axs[0].plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
        index += 1

    axs[0].set_title(f'Clustering (Number of Clusters: {best_n_clusters})')
    axs[0].legend(prop={'family': 'MS Gothic'}, loc='upper center', bbox_to_anchor=(.5, -.15))

    # 凡例を描画（subplot(2,1,2)）
    axs[1].axis('off')
    legend_labels = [str(value) + '(' + str(labels_xmeans[index]) + ')' for index, value in enumerate(grouped.T.columns)]
    axs[1].legend(labels=legend_labels, prop={'family': 'MS Gothic'})

    plt.tight_layout()

    return fig





    fig, axs = plt.subplots(2,1,1)
    index = 0
    for value in grouped.T.columns:
        axs.plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
        index += 1

    axs.set_title(f'Clustering (Number of Clusters: {best_n_clusters})')
    axs.legend(bbox_to_anchor=(1, 1), loc='upper left',  prop={'family':'MS Gothic'})
    plt.tight_layout()

# クラスタリングの結果をプロットする関数
    cmap = plt.get_cmap('tab10')  # 色の指定
    fig, axs = plt.subplots(2,1,2)

    # 凡例を作成
    legend_labels = [str(value) + '(' + str(labels_xmeans[index]) + ')' for index, value in enumerate(grouped.T.columns)]
    axs.legend(labels=legend_labels, prop={'family': 'MS Gothic'})
    axs.set_title(f'Clustering (Number of Clusters: {best_n_clusters})')

    plt.tight_layout()
    fig = plt.gcf()  # 現在のFigureを取得

    return fig



    # エルボー法
    sum_of_squared_errors = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=0, init='random')
        model.fit(grouped)
        sum_of_squared_errors.append(model.inertia_)  # 損失関数の値を保存

    plt.plot(range(1, 11), sum_of_squared_errors, marker='o')
    plt.xlabel('number of clusters')
    plt.ylabel('sum of squared errors')
    plt.show()


# In[73]:


#X-means法（シルエット分析（＝最も高いシルエットスコアを持つクラスタ数が最適なクラスタ数とする方法））
#シルエットスコア = クラスタ内のデータの密度や分離度を表す

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# データ読み込み（日付を解釈してインデックスに設定する）
df = pd.read_csv('tourists.csv', parse_dates=['date'], index_col='date')

# 欠損値を処理（均値で置換）
df = df.fillna(df.mean())
# 商品名を行、日付を列に持つデータセットに変換
grouped = df.T

# tslearn用のデータセットに変換
time_np = to_time_series_dataset(grouped)

# クラスタ数の範囲を指定
min_clusters = 2
max_clusters = 10

# シルエットスコアを最大化するクラスタ数を探索
best_score = -1
best_n_clusters = min_clusters
for n_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(grouped)
    silhouette_avg = silhouette_score(time_np.reshape((len(time_np), -1)), labels)
    
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

# 最適なクラスタ数を使用してX-means法を適用
xmeans = TimeSeriesKMeans(n_clusters=best_n_clusters, metric='euclidean', verbose=True,random_state=0, n_init=1, max_iter=100)
labels_xmeans = xmeans.fit_predict(grouped)
#print(labels_xmeans)

# クラスタリングの結果をプロット
#色の指定
cmap = plt.get_cmap('tab10')
fig = plt.figure(figsize=(20, 8))
index = 0
for value in grouped.T.columns:
    plt.plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
    index += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.title(f'Clustering (Number of Clusters: {best_n_clusters})')
plt.show()


# In[74]:


#kmeansクラスタリング

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# データ読み込み（日付を解釈してインデックスに設定する）
df = pd.read_csv('tourists.csv', parse_dates=['date'], index_col='date')

# 欠損値を処理（均値で置換）
df = df.fillna(df.mean())
# 商品名を行、日付を列に持つデータセットに変換
grouped = df.T

# tslearn用のデータセットに変換
time_np = to_time_series_dataset(grouped)

# 全体のプロット
fig, ax = plt.subplots(figsize=(20, 8))

for i, x in enumerate(time_np[:]):
    ax.plot(x, label=grouped.index[i])

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.show()

#クラスタ数が適切かの確認（グラフが曲がる瞬間の値が適切）
inertia = []
for n_clusters in range(1,12):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean')
    km.fit(time_np)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1,12), inertia, marker='o')
plt.xlabel('cluster value')
plt.ylabel('SSE')
plt.title('elbow method')
plt.show()

n = 4 #クラスタリング設定数
xmeans = TimeSeriesKMeans(n_clusters=n, metric='euclidean', verbose=True, random_state=0, n_init=1, max_iter=100)
labels_xmeans = xmeans.fit_predict(grouped)
#print(labels_xmeans)

# クラスタリングの結果をプロット
cmap = plt.get_cmap('tab10')
fig = plt.figure(figsize=(20, 8))
index = 0
for value in grouped.T.columns:
    plt.plot(grouped.T.index, grouped.T[value], label=str(value)+'('+str(labels_xmeans[index])+')', color=cmap(labels_xmeans[index]))
    index += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.title('Clustering')
plt.show()'''