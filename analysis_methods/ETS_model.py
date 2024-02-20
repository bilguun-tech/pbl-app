# ETS (AAA) Model, 1つ選択

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from pandas.tseries.offsets import DateOffset


def ETS_model(original_df, column_name):
    df = original_df.copy()
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
        df["index"] = pd.to_datetime(df["date"])
        df.set_index("index", inplace=True)
        data = df[column_name].dropna()  # NaNの行を削除
        num = 7

    else:
        msg = "This data set is not predictable. Please select another analysis method."
        fig = None
        return fig, msg

    # ETSモデルの構築
    ETS_model = ETSModel(
        data, error="add", trend="add", seasonal="add", seasonal_periods=num
    )
    ETS_fit = ETS_model.fit()

    # 予測結果の取得
    pred_start = data.index[0]
    pred_end = pred_start + pd.DateOffset(days=len(data) + len(data) // 5)
    pred = ETS_fit.get_prediction(start=pred_start, end=len(data) + len(data) // 5)
    df_pred = pred.summary_frame(alpha=0.05)
    print(df_pred)

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    ## 95%信頼区間表示
    if xlabel_name == "date":
        pred_index = pd.date_range(start=pred_start, end=pred_end)
        df_pred.index = pred_index
        data.plot(label="data", color="#ff7f0e", zorder=5)

    df_pred["mean"].plot(label="mean prediction")
    df_pred["pi_lower"].plot(linestyle="--", color="tab:blue", label="95% interval")
    df_pred["pi_upper"].plot(linestyle="--", color="tab:blue", label="_")

    if xlabel_name == "Year":
        pred.endog.plot(label="data")

    plt.axhline(y=0, xmin=0, xmax=3000, color="black", linewidth=1)  # y=0の直線
    ax.ticklabel_format(style="plain", axis="y")  # 指数表記から普通の表記に変換
    plt.xlabel(xlabel_name)
    ax.set_title("ETSmodel Plot")
    ax.legend()

    error_msg = "No Error"
    return fig, error_msg

