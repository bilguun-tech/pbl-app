# ETS (AAA) Model, 1つ選択

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from pandas.tseries.offsets import DateOffset

def ETS_model(df, column_name):
    xlabel_name = df.columns[0]
    # 準備
    if xlabel_name == "年":  # 年ごとのデータの場合
        xlabel_name = "Year"
        df["index"] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01
        df.set_index("index", inplace=True)
        data = df[column_name].dropna()  # NaNの行を削除
        if len(data) >= 24:
            num = 12
        else:
            num = len(data) // 2

    elif xlabel_name == "date":  # 1日ごとの場合
        df["index"] = pd.to_datetime(df['date'])
        df.set_index("index", inplace=True)
        data = df[column_name].dropna()  # NaNの行を削除
        num = 7

    # ETSモデルの構築
    ETS_model = ETSModel(data, error="add", trend="add", seasonal="add", seasonal_periods=num)
    ETS_fit = ETS_model.fit()

    # # ETS_fitオブジェクトからモデルオブジェクトを取得
    # model = ETS_fit.model
    # # モデルオブジェクトのインデックス関連の属性を調査
    # print("属性：",model._index)
    # print("T/F：", model._index_generated)

    # 予測結果の取得
    pred_start = data.index[0]
    pred_end = pred_start + pd.DateOffset(days=len(data) + len(data) // 5)
    # pred_index = pd.Index(pred_index)
    # pred_index = pd.date_range(start=pred_start, periods=len(data) + len(data) // 5, freq=pd.infer_freq(data.index))
    pred = ETS_fit.get_prediction(start=pred_start, end=len(data) + len(data)//5)
    df_pred = pred.summary_frame(alpha=0.05)
    print(df_pred)

    ## 旧
    # if xlabel_name == "Year":
    #     pred_end = len(data) + len(data)//3 # サンプル内+1/3予測

    # else:
    #     # pred_end = data.index[-1] + pd.DateOffset(day=31)
    #     if pred_end not in data.index:
    #         pred_end = data.index[-1] # サンプル内予測

    # pred = ETS_fit.get_prediction(start=data.index[0], end=pred_end)
    # df_pred = pred.summary_frame(alpha=0.05)
    
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    ## 95%信頼区間表示
    if xlabel_name == "date":
        pred_index = pd.date_range(start=pred_start, end=pred_end) 
        df_pred.index = pred_index
        data.plot(label="data", color='#ff7f0e', zorder=5)

    df_pred["mean"].plot(label="mean prediction")
    df_pred["pi_lower"].plot(linestyle="--", color="tab:blue", label="95% interval")
    df_pred["pi_upper"].plot(linestyle="--", color="tab:blue", label="_")

    if xlabel_name == "Year":
        pred.endog.plot(label="data")

    plt.axhline(y=0, xmin=0, xmax=3000, color="black", linewidth=1) # y=0の直線
    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.xlabel(xlabel_name)
    ax.set_title("ETSmodel Plot")
    ax.legend()

    return fig