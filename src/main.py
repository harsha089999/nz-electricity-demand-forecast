"""Day-ahead hourly electricity demand forecasting on real PJM East data.

Data: PJM Interconnection, PJM East region hourly load in megawatts (2002 to Aug 2018), published on Kaggle as
robikscube/hourly-energy-consumption (PJME_hourly.csv). Stored in data/PJME_hourly.csv.gz.

Every feature is known at least 24 hours ahead (calendar plus lags of 1 day or more), so the model is a fair
day-ahead forecast, not a one-step-ahead one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "PJME_hourly.csv.gz"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
TRAIN_START, TRAIN_END, TEST_START = "2012-01-01", "2017-07-01", "2017-08-01"  # July held back as a gap


def load():
    raw = pd.read_csv(DATA, parse_dates=["Datetime"]).rename(columns={"Datetime": "ts", "PJME_MW": "demand_mw"})
    dupes = raw.ts.duplicated().sum()
    # daylight-saving changes create duplicate hours; average them
    df = raw.groupby("ts", as_index=False).demand_mw.mean().set_index("ts").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = len(full) - len(df)
    df = df.reindex(full).interpolate(limit=3)  # fill short gaps only
    df.index.name = "ts"
    print(f"Raw rows {len(raw):,}; duplicate timestamps averaged: {dupes}; missing hours filled: {missing}")
    return df.reset_index()


def features(df):
    df = df.copy()
    hol = USFederalHolidayCalendar().holidays(df.ts.min(), df.ts.max())
    df["hour"] = df.ts.dt.hour
    df["dow"] = df.ts.dt.dayofweek
    df["month"] = df.ts.dt.month
    df["doy"] = df.ts.dt.dayofyear
    df["holiday"] = df.ts.dt.normalize().isin(hol).astype(int)
    df["lag_1d"] = df.demand_mw.shift(24)
    df["lag_7d"] = df.demand_mw.shift(168)
    df["roll_mean_1d_lagged"] = df.demand_mw.shift(24).rolling(24).mean()
    df["roll_max_1d_lagged"] = df.demand_mw.shift(24).rolling(24).max()
    return df.dropna()


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)


def main():
    df = features(load())
    feats = ["hour", "dow", "month", "doy", "holiday", "lag_1d", "lag_7d", "roll_mean_1d_lagged", "roll_max_1d_lagged"]
    train = df[(df.ts >= TRAIN_START) & (df.ts < TRAIN_END)]
    test = df[df.ts >= TEST_START]
    print(f"Train {train.ts.min():%Y-%m-%d} to {train.ts.max():%Y-%m-%d} ({len(train):,} h); "
          f"test {test.ts.min():%Y-%m-%d} to {test.ts.max():%Y-%m-%d} ({len(test):,} h)\n")

    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0)
    model.fit(train[feats], train.demand_mw)
    test = test.assign(pred_gbm=model.predict(test[feats]).round(1),
                       pred_naive_1d=test.lag_1d, pred_naive_7d=test.lag_7d)

    print(f"Naive (same hour yesterday) MAPE: {mape(test.demand_mw, test.pred_naive_1d):.2f}%")
    print(f"Naive (same hour last week) MAPE: {mape(test.demand_mw, test.pred_naive_7d):.2f}%")
    print(f"Gradient boosting MAPE:           {mape(test.demand_mw, test.pred_gbm):.2f}%")

    t = test.assign(ape=(test.demand_mw - test.pred_gbm).abs() / test.demand_mw * 100)
    by_hour = t.groupby("hour").ape.mean().round(2).rename("mape_pct").reset_index()
    by_month = (t.groupby(t.ts.dt.to_period("M").astype(str).rename("month")).ape.mean().round(2)
                .rename("mape_pct").reset_index())
    by_hour.to_csv(OUT / "error_by_hour.csv", index=False)
    by_month.to_csv(OUT / "error_by_month.csv", index=False)
    test[["ts", "demand_mw", "pred_naive_1d", "pred_naive_7d", "pred_gbm", "holiday"]].to_csv(
        OUT / "predictions.csv.gz", index=False)
    print("\nError by month:\n" + by_month.to_string(index=False))
    worst_days = t.groupby(t.ts.dt.date).ape.mean().nlargest(5).round(1)
    print("\nWorst 5 days (mean APE %):\n" + worst_days.to_string())


if __name__ == "__main__":
    main()
