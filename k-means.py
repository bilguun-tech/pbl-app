import pandas as pd
import numpy as np 
from tslearn.metrics import dtw 
import matplotlib.pyplot as plt 
from tslearn.clustering import TimeSeriesKMeans

distortions = [] 

filename = "datasets/tourists.csv"
data = pd.read_csv(filename)
df = pd.DataFrame(data)

# 空白をNaNに変換する
df.replace('', pd.NA, inplace=True)
# NaNを処理する（例えば、0に置き換える）
df.fillna(0, inplace=True)
# DataFrameを表示する
print(df)

#エルボー法
#for i in range(1,11): 
#    ts_km = TimeSeriesKMeans(n_clusters=i,metric="dtw",random_state=42) 
#    ts_km.fit_predict(df)
#    distortions.append(ts_km.inertia_) 

#plt.plot(range(1,11),distortions,marker="o") 
#plt.xticks(range(1,11)) 
#plt.xlabel("date") 
#plt.ylabel("Distortion") 
#plt.show()

ts_km = TimeSeriesKMeans(n_clusters=4,metric="dtw",random_state=42) 
y_pred = ts_km.fit_predict(df)

for cluster_id in range(ts_km.n_clusters):
    cluster_points = df[y_pred == cluster_id]
    for ts in cluster_points.values:
        plt.plot(ts, color=plt.cm.jet(cluster_id / ts_km.n_clusters))

plt.title("Time Series KMeans Clustering")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()