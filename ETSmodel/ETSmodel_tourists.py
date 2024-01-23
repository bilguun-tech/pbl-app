# ETS (AAA) Model

import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# データをdfに読み込み。
df = pd.read_csv('tourists.csv', encoding='utf-8-sig')

# 国・地域を指定
country = input("国・地域名を入力してください：")

# 準備
df['date'] = pd.to_datetime(df['年'], format='%Y') #2000->2000-01-01
df.set_index('date', inplace=True)

# ETSモデルの構築
ETS_model = ETSModel(df[country], error='add', trend='add', seasonal='add', seasonal_periods=12) 
ETS_fit = ETS_model.fit()

# 予測結果の取得
pred = ETS_fit.get_prediction(start="1990", end="2032")
df_pred = pred.summary_frame(alpha=0.05)
print(df_pred)

# 可視化
plt.figure(figsize=(10, 6))
## 95%信頼区間表示
df_pred['mean'].plot(label="mean prediction")
df_pred['pi_lower'].plot(linestyle="--", color="tab:blue", label="95% interval")
df_pred['pi_upper'].plot(linestyle="--", color="tab:blue", label="_")

pred.endog.plot(label='data')
plt.ticklabel_format(style='plain',axis='y') # 指数表記から普通の表記に変換
plt.title('ETS Model Forecast')
plt.xlabel('Year')
plt.ylabel('Tourist Numbers')
plt.legend()
plt.show()



