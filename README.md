# Demand Forecasting and Inventory Optimization

## Problem

Forecast daily sales for each store-item series in the Kaggle Store Item Demand Forecasting Challenge data, then allocate constrained item supply across stores.

## Approach

`01_features.py` reads and validates `date`, `store`, `item`, and `sales` with Spark, builds calendar, US federal holiday, lag, and shifted rolling-mean features, then converts the result to pandas and saves `outputs/features.parquet`. The final 90 calendar days form the test window. Demand-history features use lags of at least 91 days.

`02_model.py` compares a 364-day seasonal-naive baseline with LightGBM. It chooses between two LightGBM configurations using a 90-day validation slice immediately before the test window, refits the selected configuration on all pre-test data, logs baseline, tuning, and final runs to MLflow, and reports held-out WAPE and RMSE. MLflow metadata uses SQLite at `mlruns/mlflow.db`; artifacts are stored under `mlruns/artifacts`.

`03_inventory.py` computes residual-based safety stock and reorder points, then uses PuLP/CBC to minimize a tangent-envelope approximation of normal expected shortage. It compares actual unmet test-window demand with proportional-to-forecast and equal-split allocations using the same item supply.

## Assumptions

- `ASSUMPTION_LEAD_TIME_DAYS = 7` days.
- `ASSUMPTION_SERVICE_LEVEL = 0.95`; the safety-stock z-score is calculated from the standard normal distribution.
- `ASSUMPTION_SUPPLY_FRACTION = 0.8` of each item's total forecast demand over the test window.
- `ASSUMPTION_DAILY_ERRORS_INDEPENDENT = True` for aggregating daily residual standard deviation over the forecast horizon.
- US holidays are US federal holidays, including observed dates.

## Results

| Measure | Result |
| --- | ---: |
| Seasonal-naive WAPE | 0.15285170205608586 |
| Seasonal-naive RMSE | 10.897469634940235 |
| LightGBM WAPE | 0.10993207669432929 |
| LightGBM RMSE | 7.792493187285158 |
| Total assumed inventory supply | 1,980,559 |
| PuLP actual unmet demand | 478,490 |
| Proportional-to-forecast actual unmet demand | 477,850 |
| Equal-split actual unmet demand | 527,645 |

## Limitations

- Lead time, target service level, and supply fraction are assumptions (`7` days, `0.95`, and `0.8` respectively), not values from the data.
- Expected shortage assumes daily errors are independent.
- Residual standard deviation is estimated from the same final 90-day test window used for actual-demand evaluation.
- This is a simulated inventory scenario using public Kaggle data, not company data.
- The equal-split benchmark divides item supply evenly across stores and can exceed a particular store's forecast demand; it remains constrained by total item supply.

## Screenshots

- [forecast_store_1_item_1.png](screenshots/forecast_store_1_item_1.png)
- [mlflow_runs.png](screenshots/mlflow_runs.png)

## Run

Requirements: Python 3.13, Java installed, and `JAVA_HOME` set to the JDK installation directory. The verified local run used JDK 23 and PySpark 4.2.0. Spark may print a missing-`winutils.exe` warning on Windows; this workflow avoids Spark filesystem writes and saves Parquet with pandas/PyArrow. Set `DATA_PATH` and `OUTPUT_PATH` at the top of the scripts if the files or output location differ.

In PowerShell from this folder, first set `JAVA_HOME` to your installed JDK home directory. The commands below expect `JAVA_HOME` to be set:

```powershell
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
py -3.13 -m venv .venv
& '.venv/Scripts/python.exe' -m pip install -r requirements.txt
& '.venv/Scripts/python.exe' 01_features.py
& '.venv/Scripts/python.exe' 02_model.py
& '.venv/Scripts/python.exe' 03_inventory.py
```

Start the MLflow UI in a separate PowerShell window from this folder:

```powershell
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
& '.venv/Scripts/python.exe' -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open `http://127.0.0.1:5000`. Outputs are written under `outputs/`.