import os
import sqlite3

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

RAW_DATA_PATH = "train.csv"
OUTPUT_PATH = "outputs"
POWERBI_PATH = "outputs/powerbi"
DATABASE_PATH = "outputs/powerbi/demand.db"
ABC_CUTOFFS = (0.8, 0.95)

output_path = os.path.abspath(OUTPUT_PATH)
powerbi_path = os.path.abspath(POWERBI_PATH)
os.makedirs(powerbi_path, exist_ok=True)

sales = pd.read_csv(os.path.abspath(RAW_DATA_PATH), parse_dates=["date"])
predictions = pd.read_parquet(os.path.join(output_path, "predictions.parquet"))
allocation = pd.read_csv(os.path.join(output_path, "inventory_allocation.csv"))
reorder_points = pd.read_csv(os.path.join(output_path, "reorder_points.csv"))
model_metrics = pd.read_csv(os.path.join(output_path, "model_metrics.csv"))

fact_sales = sales[["date", "store", "item", "sales"]].sort_values(["date", "store", "item"])

fact_forecast = predictions[["date", "store", "item", "actual", "forecast", "baseline_forecast"]].copy()
fact_forecast["date"] = pd.to_datetime(fact_forecast["date"])
fact_forecast["forecast"] = fact_forecast["forecast"].round(3)
fact_forecast["absolute_error"] = (fact_forecast["actual"] - fact_forecast["forecast"]).abs().round(3)
fact_forecast["baseline_absolute_error"] = (fact_forecast["actual"] - fact_forecast["baseline_forecast"]).abs()

fact_inventory = allocation.merge(
    reorder_points[["store", "item", "forecast_mean_daily_demand", "residual_std_daily", "safety_stock", "reorder_point"]],
    on=["store", "item"],
    how="left",
)

date_range = pd.date_range(fact_sales["date"].min(), fact_sales["date"].max(), freq="D")
holidays = set(USFederalHolidayCalendar().holidays(start=date_range.min(), end=date_range.max()))
test_start = fact_forecast["date"].min()
dim_date = pd.DataFrame({"date": date_range})
dim_date["year"] = dim_date["date"].dt.year
dim_date["quarter"] = "Q" + dim_date["date"].dt.quarter.astype(str)
dim_date["month_number"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.strftime("%b")
dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
dim_date["month_start"] = dim_date["date"].dt.to_period("M").dt.to_timestamp()
dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
dim_date["day_of_week_number"] = dim_date["date"].dt.dayofweek + 1
dim_date["day_of_week_name"] = dim_date["date"].dt.strftime("%a")
dim_date["is_weekend"] = dim_date["day_of_week_number"].isin([6, 7]).astype(int)
dim_date["is_us_holiday"] = dim_date["date"].isin(holidays).astype(int)
dim_date["period"] = (dim_date["date"] >= test_start).map({True: "Forecast window", False: "History"})

dim_store = pd.DataFrame({"store": sorted(fact_sales["store"].unique())})
dim_store["store_name"] = "Store " + dim_store["store"].astype(str)

# ABC classification: A items make up the first 80% of total sales, B the next 15%, C the rest.
item_sales = fact_sales.groupby("item", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
item_sales["cumulative_share"] = item_sales["sales"].cumsum() / item_sales["sales"].sum()
item_sales["abc_class"] = pd.cut(
    item_sales["cumulative_share"].shift(fill_value=0),
    bins=[-0.01, ABC_CUTOFFS[0], ABC_CUTOFFS[1], 1.0],
    labels=["A", "B", "C"],
).astype(str)
dim_item = item_sales[["item", "abc_class"]].sort_values("item").reset_index(drop=True)
dim_item["item_name"] = "Item " + dim_item["item"].astype(str)
dim_item = dim_item[["item", "item_name", "abc_class"]]

tables = {
    "fact_sales": fact_sales,
    "fact_forecast": fact_forecast,
    "fact_inventory": fact_inventory,
    "dim_date": dim_date,
    "dim_store": dim_store,
    "dim_item": dim_item,
    "model_metrics": model_metrics,
}
for table_name, table in tables.items():
    table.to_csv(os.path.join(powerbi_path, f"{table_name}.csv"), index=False, date_format="%Y-%m-%d")

if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
with sqlite3.connect(os.path.abspath(DATABASE_PATH)) as connection:
    for table_name, table in tables.items():
        sqlite_table = table.copy()
        for column in sqlite_table.select_dtypes(include="datetime").columns:
            sqlite_table[column] = sqlite_table[column].dt.strftime("%Y-%m-%d")
        sqlite_table.to_sql(table_name, connection, index=False)

for table_name, table in tables.items():
    print(f"{table_name}: {len(table):,} rows")
print(f"Power BI files written to {powerbi_path}")
