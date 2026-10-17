"""Half-hourly electricity demand forecasting with lag/calendar/weather features."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
HOLIDAYS = pd.to_datetime(["2025-12-25", "2025-12-26", "2026-01-01", "2026-01-02", "2026-02-06",
                           "2026-04-03", "2026-04-06", "2026-04-27", "2026-06-01", "2026-07-10",
                           "2026-10-26", "2025-11-14"])


def make_data(seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-10-01", "2026-10-15 23:30", freq="30min")
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    temp = 11 - 6 * np.cos((doy - 20) / 365 * 2 * np.pi) + 4 * np.sin((hour - 9) / 24 * 2 * np.pi) \
        + rng.normal(0, 1.5, len(ts))
    daily = 300 * np.exp(-((hour - 8) ** 2) / 4) + 380 * np.exp(-((hour - 18.5) ** 2) / 5)
    weekend = np.isin(ts.dayofweek, [5, 6]) | ts.normalize().isin(HOLIDAYS)
    heating = 35 * np.clip(14 - temp, 0, None)
    demand = 1600 + daily * np.where(weekend, 0.75, 1.0) + heating + rng.normal(0, 30, len(ts))
    return pd.DataFrame({"ts": ts, "temp_c": temp.round(1), "demand_mw": demand.round(1)})


def features(df):
    df = df.copy()
    df["hh"] = df.ts.dt.hour * 2 + df.ts.dt.minute // 30
    df["dow"] = df.ts.dt.dayofweek
    df["month"] = df.ts.dt.month
    df["holiday"] = df.ts.dt.normalize().isin(HOLIDAYS).astype(int)
    df["heating_deg"] = np.clip(14 - df.temp_c, 0, None)
    df["lag_1d"] = df.demand_mw.shift(48)
    df["lag_7d"] = df.demand_mw.shift(336)
    df["roll_mean_1d_lagged"] = df.demand_mw.shift(48).rolling(48).mean()
    return df.dropna()


def mape(y, p):
    return float(np.mean(np.abs(y - p) / y) * 100)


def main():
    df = features(make_data())
    feats = ["hh", "dow", "month", "holiday", "temp_c", "heating_deg", "lag_1d", "lag_7d", "roll_mean_1d_lagged"]
    train = df[df.ts < "2026-08-01"]
    test = df[df.ts >= "2026-09-01"]  # August held back as a gap/validation month

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    model.fit(train[feats], train.demand_mw)
    test = test.assign(pred_gbm=model.predict(test[feats]).round(1), pred_naive=test.lag_7d)

    print(f"Naive (same time last week) MAPE: {mape(test.demand_mw, test.pred_naive):.2f}%")
    print(f"Gradient boosting MAPE:            {mape(test.demand_mw, test.pred_gbm):.2f}%")

    by_hour = (test.assign(hour=test.ts.dt.hour, ape=(test.demand_mw - test.pred_gbm).abs() / test.demand_mw * 100)
               .groupby("hour").ape.mean().round(2).rename("mape_pct").reset_index())
    by_hour.to_csv(OUT / "error_by_hour.csv", index=False)
    test[["ts", "demand_mw", "pred_naive", "pred_gbm", "temp_c", "holiday"]].to_csv(OUT / "predictions.csv", index=False)
    print("Worst hours:", by_hour.nlargest(3, "mape_pct").to_dict("records"))


if __name__ == "__main__":
    main()
