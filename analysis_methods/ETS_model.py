# ETS (AAA) Model, 1つ選択

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def ETS_model(df, column_name):
    # 準備
    if "年" in df.columns:  # 年ごとのデータの場合
        xlabel_name = "Year"
        df["date"] = pd.to_datetime(df.iloc[:, 0], format="%Y")  # 2000->2000-01-01
        df.set_index("date", inplace=True)
        data = df[column_name].dropna()  # NaNの行を削除
        if len(data) >= 24:
            num = 12
        else:
            num = len(data) // 2

    elif "date" in df.columns:  # 1日ごとの場合
        xlabel_name = "date"
        df.set_index("date", inplace=True)
        data = df[column_name].dropna()  # NaNの行を削除
        num = 7

    # ETSモデルの構築
    ETS_model = ETSModel(
        data, error="add", trend="add", seasonal="add", seasonal_periods=num
    )
    ETS_fit = ETS_model.fit()

    # 予測結果の取得
    # pred = ETS_fit.get_prediction(start = data.index[0], end = data.index[-1]) # サンプル内予測
    pred = ETS_fit.get_prediction(start = data.index[0], end = len(data)+len(data)//3) # サンプル+1/3予測
    df_pred = pred.summary_frame(alpha=0.05)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    ## 95%信頼区間表示
    df_pred["mean"].plot(label="mean prediction")
    df_pred["pi_lower"].plot(linestyle="--", color="tab:blue", label="95% interval")
    df_pred["pi_upper"].plot(linestyle="--", color="tab:blue", label="_")
    pred.endog.plot(label="data")
    plt.axhline(y=0, xmin=0, xmax=3000, color="black", linewidth=1)
    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.xlabel(xlabel_name)
    ax.set_title("ETSmodel Plot")
    ax.legend()

    return fig