# Data

Real data, not simulated. Files are the original CSVs, gzip-compressed to keep the repo small. pandas and DuckDB read
`.csv.gz` directly, so nothing needs unzipping.

**Dataset:** PJM Interconnection hourly load, PJM East region

- [Kaggle: Hourly Energy Consumption (robikscube)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- [PJM Data Miner (original source)](https://dataminer2.pjm.com/)

**Licence:** Published on Kaggle under CC0 (public domain).

| File | Contents |
|---|---|
| `PJME_hourly.csv.gz` | 145,366 hourly readings of PJM East load in megawatts, 2002-01-01 to 2018-08-03 |

To refresh from Kaggle instead, download the dataset with the Kaggle CLI (`kaggle datasets download <owner/dataset>`),
then gzip the needed files into this folder with the same names.
