

## ライブラリの設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import matplotlib.dates as mdates
from IPython.display import display


from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import itertools
import statistics
## データの読み込み
data_dir = Path("data")
print(data_dir)
file_names = [
    "ETTh1.csv",
    "ETTh2.csv",
    "ETTm1.csv",
    "ETTm2.csv"
]

datasets = {}

##EDA
# データの全体像の把握
for file_name in file_names:
    file_path = data_dir / file_name
    dataset_name = file_path.stem

    datasets[dataset_name] = pd.read_csv(file_path)

    print(f"\n===== {dataset_name} =====")
    print("ファイル:", file_path)
    print("データサイズ:", datasets[dataset_name].shape)
    display(datasets[dataset_name].head())

for name, df in datasets.items():
    print(f"\n===== {name} =====")
    print(df.info())
    print("欠損値数:")
    print(df.isnull().sum())
    print("重複行数:", df.duplicated().sum())
    print("期間:", df["date"].min(), "～", df["date"].max())

#時系列グラフ
# dateを日時型に変換し、日時順に並べる
for name, df in datasets.items():
    df["date"] = pd.to_datetime(df["date"])
    datasets[name] = df.sort_values("date").reset_index(drop=True)

# 1時間間隔のデータのみを使用
weekly_h1 = (
    datasets["ETTh1"]
    .set_index("date")["OT"]
    .resample("7D")
    .mean()
)

weekly_h2 = (
    datasets["ETTh2"]
    .set_index("date")["OT"]
    .resample("7D")
    .mean()
)

#一週間の平均温度推移
plt.figure(figsize=(15, 6))

plt.plot(
    weekly_h1.index,
    weekly_h1.values,
    label="Transformer 1",
    linewidth=1.5
)

plt.plot(
    weekly_h2.index,
    weekly_h2.values,
    label="Transformer 2",
    linewidth=1.5
)

plt.title("Weekly Average Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Weekly Average OT")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#１時間ごとの１日分の温度推移（季節別）
target_dates = [
    "2017-01-01",
    "2017-04-01",
    "2017-07-01",
    "2017-10-01"
]


fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16, 10)
)

axes = axes.flatten()


for ax, target_date in zip(axes, target_dates):

    start = pd.Timestamp(target_date)
    end = start + pd.Timedelta(days=1)

    h1_one_day = datasets["ETTh1"][
        (datasets["ETTh1"]["date"] >= start)
        & (datasets["ETTh1"]["date"] < end)
    ]

    h2_one_day = datasets["ETTh2"][
        (datasets["ETTh2"]["date"] >= start)
        & (datasets["ETTh2"]["date"] < end)
    ]

    ax.plot(
        h1_one_day["date"],
        h1_one_day["OT"],
        marker="o",
        label="Transformer 1"
    )

    ax.plot(
        h2_one_day["date"],
        h2_one_day["OT"],
        marker="o",
        label="Transformer 2"
    )

    ax.set_title(target_date)
    ax.set_xlabel("Time")
    ax.set_ylabel("OT")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=3)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )

    ax.tick_params(axis="x", rotation=45)


fig.suptitle(
    "Hourly Oil Temperature by Season",
    fontsize=16
)

plt.tight_layout()
plt.show()

#１５分ごとの１日分の温度推移。（季節別）
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16, 10)
)

axes = axes.flatten()


for ax, target_date in zip(axes, target_dates):

    start = pd.Timestamp(target_date)
    end = start + pd.Timedelta(days=1)

    m1_one_day = datasets["ETTm1"][
        (datasets["ETTm1"]["date"] >= start)
        & (datasets["ETTm1"]["date"] < end)
    ]

    m2_one_day = datasets["ETTm2"][
        (datasets["ETTm2"]["date"] >= start)
        & (datasets["ETTm2"]["date"] < end)
    ]

    ax.plot(
        m1_one_day["date"],
        m1_one_day["OT"],
        label="Transformer 1"
    )

    ax.plot(
        m2_one_day["date"],
        m2_one_day["OT"],
        label="Transformer 2"
    )

    ax.set_title(target_date)
    ax.set_xlabel("Time")
    ax.set_ylabel("OT")
    ax.grid(alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=3)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )

    ax.tick_params(axis="x", rotation=45)


fig.suptitle(
    "15-Minute Oil Temperature by Season",
    fontsize=16
)

plt.tight_layout()
plt.show()


#月ごとの油温度の確認


monthly_results = []

for name in ["ETTh1", "ETTh2"]:

    df = datasets[name].copy()
    df["date"] = pd.to_datetime(df["date"])

    df_2017 = df[df["date"].dt.year == 2017].copy()
    df_2017["month"] = df_2017["date"].dt.month

    monthly_ot = (
        df_2017
        .groupby("month")["OT"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    monthly_ot["dataset"] = name
    monthly_results.append(monthly_ot)


monthly_df = pd.concat(
    monthly_results,
    ignore_index=True
)
print("月ごとの油温度の確認")
print(monthly_df)

#月平均油温推移のグラフ
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=monthly_df,
    x="month",
    y="mean",
    hue="dataset",
    marker="o"
)

plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Average Oil Temperature")
plt.title("Monthly Average Oil Temperature in 2017")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#月全体の日内変化
target_months = [1, 4, 7, 10]

daily_profile_results = []

for name in ["ETTm1", "ETTm2"]:

    df = datasets[name].copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["date"].dt.year == 2017)
        & (df["date"].dt.month.isin(target_months))
    ].copy()

    df["month"] = df["date"].dt.month

    # 15分単位の時刻を0～23.75で表す
    df["time"] = (
        df["date"].dt.hour
        + df["date"].dt.minute / 60
    )

    profile = (
        df.groupby(["month", "time"])["OT"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    profile["dataset"] = name
    daily_profile_results.append(profile)


daily_profile_df = pd.concat(
    daily_profile_results,
    ignore_index=True
)

#季節ごとの日内変化
month_names = {
    1: "January",
    4: "April",
    7: "July",
    10: "October"
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
    sharex=True
)

axes = axes.flatten()


for ax, month in zip(axes, target_months):

    month_data = daily_profile_df[
        daily_profile_df["month"] == month
    ]

    sns.lineplot(
        data=month_data,
        x="time",
        y="mean",
        hue="dataset",
        ax=ax
    )

    ax.set_title(month_names[month])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average OT")
    ax.set_xticks(range(0, 25, 3))
    ax.grid(alpha=0.3)


fig.suptitle(
    "Average Intraday Oil Temperature Profiles in 2017",
    fontsize=16
)

plt.tight_layout()
plt.show()

#ピーク時刻を数値で確認
peak_times = (
    daily_profile_df.loc[
        daily_profile_df
        .groupby(["dataset", "month"])["mean"]
        .idxmax()
    ]
    [["dataset", "month", "time", "mean"]]
    .sort_values(["dataset", "month"])
)

print(peak_times)

#相関係数の変圧器ごとの確認
numeric_columns = [
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT"
]


for name in ["ETTh1", "ETTh2"]:

    df = datasets[name]

    correlation_matrix = (
        df[numeric_columns]
        .corr()
    )

    print(f"\n===== {name}: OTとの相関 =====")

    print(
        correlation_matrix["OT"]
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.title(f"Correlation Matrix: {name}")
    plt.tight_layout()
    plt.show()



#日時の影響を取り除いて油温の変化を確認する
print("日時の影響を取り除いた場合の油温とMULL,HULLとの相関係数")
for dataset_name in ["ETTh1", "ETTh2"]:

    df = datasets[dataset_name].copy()
    df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour

    # 同じ月・同じ時刻の平均的な油温
    df["expected_OT"] = (
        df.groupby(["month", "hour"])["OT"]
        .transform("mean")
    )

    # 平均的な周期からどれだけ外れているか
    df["OT_residual"] = (
        df["OT"] - df["expected_OT"]
    )

    residual_correlations = (
        df[
            [
                "HUFL",
                "HULL",
                "MUFL",
                "MULL",
                "LUFL",
                "LULL",
                "OT_residual"
            ]
        ]
        .corr()["OT_residual"]
        .sort_values(ascending=False)
    )

    print(f"\n===== {dataset_name} =====")
    print(residual_correlations)

#MULLまたはHULLが0であることが欠損値なのか、意味のある数値なのかを確認する
#MULLまたはHULLが0の行を抽出
df = datasets["ETTh2"].copy()
df["date"] = pd.to_datetime(df["date"])
print("MULLまたはHULLが0の行を抽出する")
print(
    df.loc[
        (df["MULL"] == 0) | (df["HULL"] == 0),
        ["date", "MULL", "HULL", "OT"]
    ]
)
#0の内訳を確認
df = datasets["ETTh2"].copy()

mull_zero = df["MULL"] == 0
hull_zero = df["HULL"] == 0


print("全データ数:", len(df))

print(
    "MULLだけ0:",
    (mull_zero & ~hull_zero).sum()
)

print(
    "HULLだけ0:",
    (~mull_zero & hull_zero).sum()
)

print(
    "両方0:",
    (mull_zero & hull_zero).sum()
)

print(
    "どちらかが0:",
    (mull_zero | hull_zero).sum()
)

print(
    "MULLが0の割合:",
    f"{mull_zero.mean():.2%}"
)

print(
    "HULLが0の割合:",
    f"{hull_zero.mean():.2%}"
)

#0のときと0以外のときの油温を比較する
df["MULL_zero"] = df["MULL"] == 0
df["HULL_zero"] = df["HULL"] == 0


print("MULLが0かどうかによるOTの違い")

print(
    df.groupby("MULL_zero")["OT"]
    .agg(["count", "mean", "median", "std"])
)


print("\nHULLが0かどうかによるOTの違い")

print(
    df.groupby("HULL_zero")["OT"]
    .agg(["count", "mean", "median", "std"])
)

#月・時刻ごとの0の割合
df = datasets["ETTh2"].copy()
df["date"] = pd.to_datetime(df["date"])

df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour

df["MULL_zero"] = df["MULL"].eq(0)
df["HULL_zero"] = df["HULL"].eq(0)

print("青色が濃いほど0の割り合いが大きい")
fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 5)
)


for ax, column in zip(
    axes,
    ["MULL_zero", "HULL_zero"]
):

    zero_rate = (
        df.groupby(["month", "hour"])[column]
        .mean()
        .unstack()
    )

    sns.heatmap(
        zero_rate,
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_title(f"{column} Rate by Month and Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Month")


plt.tight_layout()
plt.show()

#日時の影響を除いても油温が低いか
df["expected_OT"] = (
    df.groupby(["month", "hour"])["OT"]
    .transform("mean")
)

df["OT_residual"] = (
    df["OT"] - df["expected_OT"]
)


print("MULLが0かどうかによる油温残差")

print(
    df.groupby("MULL_zero")["OT_residual"]
    .agg(["count", "mean", "median", "std"])
)


print("\nHULLが0かどうかによる油温残差")

print(
    df.groupby("HULL_zero")["OT_residual"]
    .agg(["count", "mean", "median", "std"])
)

##モデルの作成

#モデルの定義と評価を含めた関数
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)

    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    print(f"\n===== {dataset_name} =====")

    print("訓練期間:")
    print(
        train_df["date"].min(),
        "～",
        train_df["date"].max()
    )

    print("テスト期間:")
    print(
        test_df["date"].min(),
        "～",
        test_df["date"].max()
    )


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        "OT_lag_1",
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["OT"]

    X_test = test_df[feature_columns]
    y_test = test_df["OT"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "dataset_name":dataset_name,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )

##1時間後の温度を予測するモデルを作成する
print("1時間後の温度を予測するモデル")

#モデルの定義と評価を含めた関数
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)
   #1時間後の予測温度を記録する
    model_df["target_OT_1h"] = model_df["OT"].shift(-1)
    model_df["target_date"]=model_df["date"].shift(-1)
    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    print(f"\n===== {dataset_name} =====")

    print("訓練期間:")
    print(
        train_df["date"].min(),
        "～",
        train_df["date"].max()
    )

    print("テスト期間:")
    print(
        test_df["date"].min(),
        "～",
        test_df["date"].max()
    )


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        #１時間前ではなく、現在の情報を基に１時間後のOTを予測する
        "OT",
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["target_OT_1h"]

    X_test = test_df[feature_columns]
    y_test = test_df["target_OT_1h"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["target_date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["target_date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "dataset_name":dataset_name,
        "target_date":test_df["target_date"],
        "target_OT_1h":test_df["target_OT_1h"],
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "y_pred": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )

##ROIの計算

#予測値が閾値を超えた回数の計算
alert_results = {}

for name, result in results.items():
    y_test = np.asarray(result["y_test"])
    y_pred = np.asarray(result["y_pred"])

    # 実測値の最大値の90%を閾値にする
    alert_threshold = y_test.max() * 0.90

    # 予測値に基づくアラート
    predicted_alert = y_pred >= alert_threshold
    alert_count = predicted_alert.sum()

    # 実際に閾値を超えたか
    actual_abnormal = y_test >= alert_threshold

    # 予測も異常、実測も異常だった件数
    true_positive_count = (
        predicted_alert & actual_abnormal
    ).sum()

    alert_results[name] = {
        "alert_threshold": alert_threshold,
        "alert_count": alert_count,
        "true_positive_count": true_positive_count
    }

    print(f"===== {name} =====")
    print(f"アラート閾値: {alert_threshold:.2f}℃")
    print(f"予測によるアラート数: {alert_count}件")
    print(f"正しく異常を予測した数: {true_positive_count}件")

#予防処置費と事後処理費（段階の設定）
prevention_fee=[
    15,20,40
]

post_event_fee=[
    80,150
]

#ROIの計算
for name, alert_result in alert_results.items():

    alert_count = alert_result["alert_count"]
    true_positive_count = alert_result["true_positive_count"]

    roi_results = []

    for prevention_cost, post_event_cost in itertools.product(
        prevention_fee,
        post_event_fee
    ):
        total_prevention_cost = (
            alert_count * prevention_cost
        )

        avoided_loss = (
            true_positive_count * post_event_cost
        )

        net_benefit = (
            avoided_loss - total_prevention_cost
        )

        if total_prevention_cost > 0:
            roi = (
                net_benefit / total_prevention_cost
            ) * 100
        else:
            roi = np.nan

        roi_results.append({
            "prevention_cost": prevention_cost,
            "post_event_cost": post_event_cost,
            "total_prevention_cost": total_prevention_cost,
            "avoided_loss": avoided_loss,
            "net_benefit": net_benefit,
            "roi": roi
        })

    net_benefits = [
        result["net_benefit"]
        for result in roi_results
    ]

    roi_numbers = [
        result["roi"]
        for result in roi_results
        if not np.isnan(result["roi"])
    ]
    roi_over_100_count = sum(
    roi > 100
    for roi in roi_numbers
    )

    roi_over_100_ratio = (
      roi_over_100_count / len(roi_numbers) * 100
    )

    print(f"\n===== {name}のROI =====")
    print(f"純改善額の最高値: {max(net_benefits):,.0f}万円")
    print(
        f"純改善額の平均値: "
        f"{statistics.mean(net_benefits):,.0f}万円"
    )
    print(
        f"純改善額の中央値: "
        f"{statistics.median(net_benefits):,.0f}万円"
    )
    print(f"純改善額の最低値: {min(net_benefits):,.0f}万円")

    if roi_numbers:
        print(f"ROIの最高値: {max(roi_numbers):.1f}%")
        print(
            f"ROIの平均値: "
            f"{statistics.mean(roi_numbers):.1f}%"
        )
        print(
            f"ROIの中央値: "
            f"{statistics.median(roi_numbers):.1f}%"
        )
        print(f"ROIの最低値: {min(roi_numbers):.1f}%")
        print(f"ROIが100%を超える割合: {roi_over_100_ratio:.1f}%")
    else:
        print("アラートが0件だったためROIを計算できません。")

#24時間後の温度を予測するモデルを作成する
print("24時間後の温度を予測するモデル")

#モデルの定義と評価を含めた関数
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)
   #1時間後の予測温度を記録する
    model_df["target_OT_1h"] = model_df["OT"].shift(-24)
    model_df["target_date"]=model_df["date"].shift(-24)
    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    print(f"\n===== {dataset_name} =====")

    print("訓練期間:")
    print(
        train_df["date"].min(),
        "～",
        train_df["date"].max()
    )

    print("テスト期間:")
    print(
        test_df["date"].min(),
        "～",
        test_df["date"].max()
    )


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        #１時間前ではなく、現在の情報を基に１時間後のOTを予測する
        "OT",
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["target_OT_1h"]

    X_test = test_df[feature_columns]
    y_test = test_df["target_OT_1h"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["target_date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["target_date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )



#特定の特徴量を除外した場合の評価
#OT_lagの除外
print("特定の特徴量を除外した場合の評価")
print("OT_lagの除外")

#モデルの再定義
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)

    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["OT"]

    X_test = test_df[feature_columns]
    y_test = test_df["OT"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )

#時間の除外
print("時間の除外")


#モデルの再定義
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)

    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        "OT_lag_1",
        "elapsed_days",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["OT"]

    X_test = test_df[feature_columns]
    y_test = test_df["OT"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )

#月の除外
print("月の除外")


#モデルの再定義
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)

    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        selected_load_feature,
        "OT_lag_1",
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["OT"]

    X_test = test_df[feature_columns]
    y_test = test_df["OT"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )

#MULLとHULLの除外
print("MULLとHULLの除外")


#モデルの再定義
def train_regression_model(df, dataset_name):

# 元データを変更しないようにコピー
    model_df = df.copy()


# 前処理

#日時データ処理
    model_df["date"] = pd.to_datetime(model_df["date"])

    model_df = (
        model_df.sort_values("date")
        .reset_index(drop=True)
    )

  # 日時特徴量
    model_df["hour"] = model_df["date"].dt.hour
    model_df["dayofweek"] = model_df["date"].dt.dayofweek
    model_df["month"] = model_df["date"].dt.month

  # 周期性を三角関数により表現
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["hour"] / 24
    )

    model_df["dayofweek_sin"] = np.sin(
        2 * np.pi * model_df["dayofweek"] / 7
    )
    model_df["dayofweek_cos"] = np.cos(
        2 * np.pi * model_df["dayofweek"] / 7
    )

    model_df["month_sin"] = np.sin(
        2 * np.pi * (model_df["month"] - 1) / 12
    )
    model_df["month_cos"] = np.cos(
        2 * np.pi * (model_df["month"] - 1) / 12
    )


  # 時系列分割
   #経過日数を記録する
    model_df["elapsed_days"] = (model_df["date"] - model_df["date"].min()).dt.total_seconds() / (60 * 60 * 24)
   # 1時間前のOTの過去の影響を記録する
    model_df["OT_lag_1"] = model_df["OT"].shift(1)

    model_df = model_df.dropna()

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()


# HULL・MULLの選択

    correlation_with_ot = (
        train_df[["MULL", "HULL", "OT"]]
        .corr()["OT"]
        .drop("OT")
    )

    selected_load_feature = (
        correlation_with_ot.abs().idxmax()
    )

    print("\nOTとの相関:")
    print(correlation_with_ot)

    print(
        "採用する特徴量:",
        selected_load_feature
    )

# 説明変数・目的変数の設定

    feature_columns = [
        "OT_lag_1",
        "elapsed_days",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos"
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["OT"]

    X_test = test_df[feature_columns]
    y_test = test_df["OT"]

# モデル学習

    model = LinearRegression()

    model.fit(X_train, y_train)

# 予測

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

# 評価

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_prediction
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    print("\n訓練データ")
    print("MAE :", train_mae)
    print("RMSE:", train_rmse)
    print("R²  :", train_r2)

    print("\nテストデータ")
    print("MAE :", test_mae)
    print("RMSE:", test_rmse)
    print("R²  :", test_r2)

# 回帰係数

    coefficient_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_
    })

    print("\n回帰係数:")
    print(coefficient_df)

# 予測結果の可視化

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_df["date"],
        y_test,
        label="Actual OT",
        linewidth=1
    )

    plt.plot(
        test_df["date"],
        test_prediction,
        label="Predicted OT",
        linewidth=1
    )

    plt.title(f"{dataset_name}: Actual and Predicted OT")
    plt.xlabel("Date")
    plt.ylabel("OT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 後から使えるように結果を返す

    return {
        "model": model,
        "feature_columns": feature_columns,
        "selected_load_feature": selected_load_feature,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train ,
        "y_train":y_train ,
        "X_test":X_test ,
        "y_test":y_test ,
        "test_prediction": test_prediction,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }
#データセットの準備
transformer_datasets = {
    "transformer_1": datasets["ETTh1"].copy(),
    "transformer_2": datasets["ETTh2"].copy()
}

##モデルの評価

 #関数の呼び出し
results={
    "results_1":train_regression_model(df=transformer_datasets["transformer_1"], dataset_name="ETTh1"),
    "results_2":train_regression_model(df=transformer_datasets["transformer_2"], dataset_name="ETTh2")
}

metric_names = [
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_r2",
    "test_r2"
]

for result_name, result in results.items():

    print(f"\n===== {result_name} =====")

    for metric_name in metric_names:
        print(
            metric_name,
            ":",
            result[metric_name]
        )



