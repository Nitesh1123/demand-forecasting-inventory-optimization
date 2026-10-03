# Databricks notebook source
# COMMAND ----------
from datetime import timedelta
import os

from pandas.tseries.holiday import USFederalHolidayCalendar
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

DATA_PATH = "train.csv"
OUTPUT_PATH = "outputs"
HORIZON_DAYS = 90
MINIMUM_HISTORY_LAG_DAYS = HORIZON_DAYS + 1
LAG_DAYS = (MINIMUM_HISTORY_LAG_DAYS, 182, 364)

spark = SparkSession.builder.master("local[*]").appName("demand-forecast-features").getOrCreate()
output_path = os.path.abspath(OUTPUT_PATH)
os.makedirs(output_path, exist_ok=True)
raw = spark.read.option("header", True).option("inferSchema", False).csv(os.path.abspath(DATA_PATH))
sales = raw.select(
    F.to_date("date", "yyyy-MM-dd").alias("date"),
    F.col("store").cast("int").alias("store"),
    F.col("item").cast("int").alias("item"),
    F.col("sales").cast("double").alias("sales"),
)

null_count = sales.filter(
    F.col("date").isNull()
    | F.col("store").isNull()
    | F.col("item").isNull()
    | F.col("sales").isNull()
).count()
duplicate_count = (
    sales.groupBy("date", "store", "item")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
print(f"Null rows: {null_count}")
print(f"Duplicate store-item-date keys: {duplicate_count}")
print(f"Input row count: {sales.count()}")
if null_count or duplicate_count:
    raise ValueError("Input contains null values or duplicate store-item-date keys")

sales = sales.repartition("store", "item").sortWithinPartitions("store", "item", "date")
print(sales.orderBy("store", "item", "date").limit(20).toPandas().to_string(index=False))

date_bounds = sales.agg(F.min("date").alias("min_date"), F.max("date").alias("max_date")).first()
holiday_dates = USFederalHolidayCalendar().holidays(
    start=date_bounds.min_date,
    end=date_bounds.max_date,
)
holidays = spark.createDataFrame(
    [(holiday.to_pydatetime().date(),) for holiday in holiday_dates],
    ["holiday_date"],
)
max_date = date_bounds.max_date
test_start = max_date - timedelta(days=HORIZON_DAYS - 1)
series_window = Window.partitionBy("store", "item").orderBy("date")

features = (
    sales.withColumn("day_of_week", F.dayofweek("date"))
    .withColumn("month", F.month("date"))
    .withColumn("day_of_year", F.dayofyear("date"))
    .join(F.broadcast(holidays), sales.date == holidays.holiday_date, "left")
    .withColumn("is_us_holiday", F.when(F.col("holiday_date").isNotNull(), 1).otherwise(0))
    .drop("holiday_date")
)

for lag_days in LAG_DAYS:
    features = features.withColumn(f"lag_{lag_days}", F.lag("sales", lag_days).over(series_window))

features = features.withColumn(
    "sales_shifted_horizon",
    F.lag("sales", MINIMUM_HISTORY_LAG_DAYS).over(series_window),
)
for window_days in (7, 28, 91):
    rolling_window = series_window.rowsBetween(-window_days + 1, 0)
    features = features.withColumn(
        f"rolling_mean_{window_days}", F.avg("sales_shifted_horizon").over(rolling_window)
    )

features = (
    features.drop("sales_shifted_horizon")
    .withColumn("split", F.when(F.col("date") >= F.lit(test_start), "test").otherwise("train"))
    .dropna(
        subset=[
            "lag_91",
            "lag_182",
            "lag_364",
            "rolling_mean_7",
            "rolling_mean_28",
            "rolling_mean_91",
        ]
    )
)

assert min(LAG_DAYS) >= HORIZON_DAYS + 1
print(f"Date range: {date_bounds.min_date} to {max_date}")
print(f"Test starts: {test_start}; test days: {HORIZON_DAYS}")
feature_count = features.count()
print(f"Feature rows after warmup: {feature_count}")
feature_pdf = features.toPandas()
feature_pdf.to_parquet(os.path.join(output_path, "features.parquet"), index=False)
print(feature_pdf.sort_values(["store", "item", "date"]).head(20).to_string(index=False))
spark.stop()