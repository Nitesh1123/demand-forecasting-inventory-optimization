# Databricks notebook source
from datetime import timedelta
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd

OUTPUT_PATH = "outputs"
DATA_PATH = "outputs/features.parquet"
HORIZON_DAYS = 90
FEATURE_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "month",
    "day_of_year",
    "is_us_holiday",
    "lag_91",
    "lag_182",
    "lag_364",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_91",
]

output_path = os.path.abspath(OUTPUT_PATH)
os.makedirs(output_path, exist_ok=True)
mlruns_path = os.path.abspath("mlruns")
os.makedirs(mlruns_path, exist_ok=True)
database_path = os.path.join(mlruns_path, "mlflow.db").replace(os.sep, "/")
mlflow.set_tracking_uri(f"sqlite:///{database_path}")
experiment_name = "store-item-demand-forecasting"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=Path(os.path.join(mlruns_path, "artifacts")).as_uri(),
    )
else:
    experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_id=experiment_id)
features = pd.read_parquet(os.path.abspath(DATA_PATH))
features["date"] = pd.to_datetime(features["date"])
test_start = features.loc[features["split"] == "test", "date"].min()
validation_start = test_start - timedelta(days=HORIZON_DAYS)

train_pdf = features.loc[
    (features["split"] == "train") & (features["date"] < validation_start), FEATURE_COLUMNS + ["sales"]
].copy()
validation_pdf = features.loc[
    (features["date"] >= validation_start) & (features["date"] < test_start),
    FEATURE_COLUMNS + ["sales"],
].copy()
test_pdf = features.loc[features["split"] == "test", ["date", *FEATURE_COLUMNS, "sales"]]
test_pdf = test_pdf.sort_values(["date", "store", "item"]).reset_index(drop=True)

def wape(actual, predicted):
    denominator = actual.sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when total actual demand is zero")
    return float(abs(actual - predicted).sum() / denominator)


def regression_metrics(actual, predicted):
    return {
        "wape": wape(actual, predicted),
        "rmse": float(((actual - predicted) ** 2).mean() ** 0.5),
    }


class SeasonalNaiveModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input["lag_364"].to_numpy()


baseline_prediction = test_pdf["lag_364"].to_numpy()
baseline_metrics = regression_metrics(test_pdf["sales"].to_numpy(), baseline_prediction)
with mlflow.start_run(run_name="seasonal_naive_364"):
    mlflow.log_param("model", "seasonal_naive")
    mlflow.log_param("seasonal_lag_days", 364)
    mlflow.log_param("test_start", str(test_start))
    mlflow.log_metrics(baseline_metrics)
    mlflow.pyfunc.log_model(artifact_path="model", python_model=SeasonalNaiveModel())

configurations = [
    {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 450},
]
tuning_results = []

for configuration_id, configuration in enumerate(configurations, start=1):
    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        **configuration,
    )
    model.fit(train_pdf[FEATURE_COLUMNS], train_pdf["sales"])
    validation_prediction = model.predict(validation_pdf[FEATURE_COLUMNS]).clip(0, None)
    metrics = regression_metrics(validation_pdf["sales"].to_numpy(), validation_prediction)
    with mlflow.start_run(run_name=f"lightgbm_validation_{configuration_id}"):
        mlflow.log_params({"model": "lightgbm", "validation_start": str(validation_start), **configuration})
        mlflow.log_metrics({f"validation_{name}": value for name, value in metrics.items()})
        mlflow.lightgbm.log_model(model, artifact_path="model")
    tuning_results.append({"configuration_id": configuration_id, **configuration, **metrics})

del train_pdf, model, validation_prediction
best_result = min(tuning_results, key=lambda result: result["wape"])
best_configuration = configurations[best_result["configuration_id"] - 1]
final_model = lgb.LGBMRegressor(
    objective="regression",
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    n_jobs=-1,
    **best_configuration,
)
full_train_pdf = features.loc[
    features["split"] == "train", FEATURE_COLUMNS + ["sales"]
].copy()
final_model.fit(full_train_pdf[FEATURE_COLUMNS], full_train_pdf["sales"])
del full_train_pdf
model_prediction = final_model.predict(test_pdf[FEATURE_COLUMNS]).clip(0, None)
model_metrics = regression_metrics(test_pdf["sales"].to_numpy(), model_prediction)

with mlflow.start_run(run_name="lightgbm_final_test"):
    mlflow.log_params(
        {"model": "lightgbm", "test_start": str(test_start), "selected_by": "validation_wape", **best_configuration}
    )
    mlflow.log_metrics(model_metrics)
    mlflow.lightgbm.log_model(final_model, artifact_path="model")

metrics_table = pd.DataFrame(
    [
        {"model": "seasonal_naive_364", **baseline_metrics},
        {"model": "lightgbm", **model_metrics},
    ]
)
tuning_table = pd.DataFrame(tuning_results)
predictions_pdf = test_pdf[["date", "store", "item", "sales"]].copy()
predictions_pdf = predictions_pdf.rename(columns={"sales": "actual"})
predictions_pdf["forecast"] = model_prediction

metrics_table.to_csv(os.path.join(output_path, "model_metrics.csv"), index=False)
tuning_table.to_csv(os.path.join(output_path, "tuning_metrics.csv"), index=False)
predictions_pdf.to_parquet(os.path.join(output_path, "predictions.parquet"), index=False)
predictions_pdf.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

print(metrics_table.to_string(index=False))
print(tuning_table.to_string(index=False))

import matplotlib.pyplot as plt

plot_directory = os.path.join(output_path, "plots")
os.makedirs(plot_directory, exist_ok=True)
pair_keys = (
    predictions_pdf[["store", "item"]].drop_duplicates().sort_values(["store", "item"]).head(3).itertuples(index=False)
)
for pair in pair_keys:
    pair_pdf = predictions_pdf[
        (predictions_pdf["store"] == pair.store) & (predictions_pdf["item"] == pair.item)
    ].sort_values("date")
    plt.figure(figsize=(10, 4))
    plt.plot(pair_pdf["date"], pair_pdf["actual"], label="Actual")
    plt.plot(pair_pdf["date"], pair_pdf["forecast"], label="LightGBM forecast")
    plt.title(f"Store {pair.store}, item {pair.item}")
    plt.xlabel("Date")
    plt.ylabel("Daily sales")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(plot_directory, f"forecast_store_{pair.store}_item_{pair.item}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
