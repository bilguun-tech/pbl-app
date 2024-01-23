import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("stock_price_AMZN01_04.csv")

df = pd.DataFrame(data)
df = df.drop("volume", axis=1)
df = df.drop("symbol", axis=1)
df.set_index("date", inplace=True)

#可視化
#df.plot()
#plt.show()


# ADF検定（原系列）定常過程かどうかを検定する
#dftest = adfuller(df)
#print('ADF Statistic: %f' % dftest[0])
#print('p-value: %f' % dftest[1])
#print('Critical values :')
#for k, v in dftest[4].items():
 #   print('\t', k, v)


#プログラムのURL:https://toukei-lab.com/python_stock


from statsmodels.tsa.arima.model import ARIMA

#ARIMAモデル データ準備
train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
train_data = train_data['close'].values
test_data = test_data['close'].values

# ARIMAモデル実装
#train_data = df["close"].values
model = ARIMA(train_data, order=(6,1,0))
model_fit = model.fit()
print(model_fit.summary())

#ARIMAモデル 予測
history = [x for x in train_data]
model_predictions = []
for time_point in range(len(test_data)):
#ARIMAモデル 実装
    model = ARIMA(history, order=(6,1,0))
    model_fit = model.fit()
#予測データの出力
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
#トレーニングデータの取り込み
    true_test_value = test_data[time_point]
    history.append(true_test_value)

#可視化
plt.plot(test_data, color='Red', label='Measured')
plt.plot(model_predictions, color='Blue', label='prediction')
plt.title('Amazon stock price prediction', fontname="MS Gothic")
plt.xlabel('date', fontname="MS Gothic")
plt.ylabel('Amazon stock price', fontname="MS Gothic")
plt.legend(prop={"family":"MS Gothic"})
plt.show()