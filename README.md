# Electricity Demand Forecast (PJM East)

[![Run project](https://github.com/harsha089999/electricity-demand-forecast/actions/workflows/run.yml/badge.svg)](https://github.com/harsha089999/electricity-demand-forecast/actions/workflows/run.yml)

**Day-ahead hourly electricity demand forecasting on 16 years of real grid data, compared honestly against naive baselines.**

> **About the data:** this project runs on a real public dataset, **PJM Interconnection hourly load, PJM East region**, included in `data/` so it runs straight after cloning. Sources and licence are listed below and in [`data/README.md`](data/README.md).

## The business question

How much electricity will the region need each hour tomorrow, and where does the forecast go wrong?

## What this project does

Grid operators and retailers buy and dispatch power a day ahead, so the forecast can only use what's known by then. This project cleans 145k hours of real load data, builds calendar and lag features that are all at least 24 hours old, trains a gradient boosting model, and checks it against 'same hour yesterday' and 'same hour last week'.

## The data

Real hourly electricity load for PJM East, which covers a large part of the US east coast, from 2002 to August 2018. The raw file has 4 duplicate timestamps (daylight saving changes), which are averaged, and 30 missing hours, which are filled by interpolation. The model trains on 2012 to June 2017, July 2017 is skipped as a gap, and it's tested on August 2017 to August 2018.

**Source:**
- [Kaggle: Hourly Energy Consumption (robikscube)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- [PJM Data Miner (original source)](https://dataminer2.pjm.com/)

**Licence:** Published on Kaggle under CC0 (public domain).

## How it works

1. Clean the raw series: average duplicate hours, reindex to a complete hourly grid, and interpolate only short gaps.
2. Features known a day ahead: hour, weekday, month, day of year, US federal holiday, load 24 hours ago, load 7 days ago, and the mean and max of the previous day.
3. Train gradient boosting (scikit-learn HistGradientBoostingRegressor) on 2012 to June 2017.
4. Test on a full year it never saw, against two naive baselines, and break the error down by hour, month and worst days.

## Run it

```bash
git clone https://github.com/harsha089999/electricity-demand-forecast
cd electricity-demand-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Outputs are written to `outputs/`. Python 3.10 or newer is recommended.

## Results from the included run

This is the actual console output when the script is run as shipped:

```text
Raw rows 145,366; duplicate timestamps averaged: 4; missing hours filled: 30
Train 2012-01-01 to 2017-06-30 (48,192 h); test 2017-08-01 to 2018-08-03 (8,809 h)

Naive (same hour yesterday) MAPE: 7.33%
Naive (same hour last week) MAPE: 11.22%
Gradient boosting MAPE:           6.04%

Error by month:
  month  mape_pct
2017-08      6.43
2017-09      7.79
2017-10      4.42
2017-11      4.13
2017-12      5.99
2018-01      7.04
2018-02      6.52
2018-03      4.12
2018-04      4.17
2018-05      6.37
2018-06      7.50
2018-07      7.97
2018-08      7.08

Worst 5 days (mean APE %):
ts
2018-07-07    30.6
2017-09-01    22.5
2018-06-03    21.1
2018-05-28    20.1
2017-09-06    18.7
```

Gradient boosting gets 6.0% MAPE, against 7.3% for 'same hour yesterday' and 11.2% for 'same hour last week'. That's a real but modest gain, and it's worth saying plainly. The error by month tells the story: spring and autumn are around 4%, while summer (June to September) and January are 7 to 8%. That's most likely heat waves and cold snaps, which a model without weather can't see coming. The worst day, 7 July 2018, had a 31% error. Adding a temperature forecast is the obvious next step.

## Output files

| File | What it contains |
|---|---|
| `outputs/predictions.csv.gz` | Every test hour: actual, both naive forecasts, model forecast and holiday flag. |
| `outputs/error_by_hour.csv` | MAPE by hour of day. |
| `outputs/error_by_month.csv` | MAPE by month of the test year. |

## Limitations and honest caveats

- No weather data, which is the biggest driver of load. This is the main reason summer error is high.
- This is a US region, not New Zealand. The same pipeline works on NZ grid demand from the Electricity Authority's EMI site; only `load()` changes.
- One train/test split. A rolling-origin backtest would give a more reliable error range.
- Point forecasts only. Operators also want a range (for example quantile regression).

## Turning it into a dashboard

Actual vs forecast line chart with a date slicer, a heatmap of error by hour and month, and a table of the worst days so analysts can check what happened on each.

## Ideas to extend it

- Join hourly temperature from NOAA and measure how much it cuts summer error.
- Quantile gradient boosting for P10/P50/P90 forecasts.
- Run the same code on NZ demand from the Electricity Authority EMI datasets.

## Skills demonstrated

Time series cleaning, leakage-free feature engineering, gradient boosting, baseline comparison, error analysis.

**Tools:** Python, pandas, scikit-learn

## Project structure

```text
electricity-demand-forecast/
├── src/main.py                 # the whole pipeline
├── data/                       # real source data (gzipped) + data/README.md with source and licence
├── outputs/                    # files from the run shown above
├── requirements.txt
├── .github/workflows/run.yml   # CI: installs and runs the project on every push
└── LICENSE
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The data in `data/` keeps its original licence (see above).
