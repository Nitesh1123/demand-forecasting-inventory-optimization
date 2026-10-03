# Databricks notebook source
import math
import os
from statistics import NormalDist

import pandas as pd
import pulp

OUTPUT_PATH = "outputs"
DATA_PATH = "outputs/predictions.parquet"
ASSUMPTION_LEAD_TIME_DAYS = 7
ASSUMPTION_SERVICE_LEVEL = 0.95
ASSUMPTION_SUPPLY_FRACTION = 0.8
available_solvers = pulp.listSolvers(onlyAvailable=True)
cbc_solver_name = next(
    (name for name in available_solvers if name == "COIN_CMD" or "CBC" in name.upper()),
    None,
)
if cbc_solver_name is None:
    raise RuntimeError('CBC solver is unavailable; install it with pip install "pulp[cbc]".')
cbc_solver = pulp.getSolver(cbc_solver_name, msg=False, gapRel=0, gapAbs=0)

output_path = os.path.abspath(OUTPUT_PATH)
os.makedirs(output_path, exist_ok=True)
predictions = pd.read_parquet(os.path.abspath(DATA_PATH))
z_score = NormalDist().inv_cdf(ASSUMPTION_SERVICE_LEVEL)
predictions["residual"] = predictions["actual"] - predictions["forecast"]
inventory_pdf = predictions.groupby(["store", "item"], as_index=False).agg(
    forecast_mean_daily_demand=("forecast", "mean"),
    forecast_demand=("forecast", "sum"),
    actual_demand=("actual", "sum"),
    residual_std_daily=("residual", "std"),
)
inventory_pdf["residual_std_daily"] = inventory_pdf["residual_std_daily"].fillna(0.0)
inventory_pdf["safety_stock"] = (
    z_score * inventory_pdf["residual_std_daily"] * math.sqrt(ASSUMPTION_LEAD_TIME_DAYS)
)
inventory_pdf["reorder_point"] = (
    inventory_pdf["forecast_mean_daily_demand"] * ASSUMPTION_LEAD_TIME_DAYS
    + inventory_pdf["safety_stock"]
)

allocation_rows = []

for item_id, item_pdf in inventory_pdf.groupby("item", sort=True):
    item_pdf = item_pdf.sort_values("store").reset_index(drop=True)
    total_forecast = float(item_pdf["forecast_demand"].sum())
    total_supply = math.floor(ASSUMPTION_SUPPLY_FRACTION * total_forecast)
    store_ids = item_pdf["store"].astype(int).tolist()
    forecast_by_store = dict(zip(store_ids, item_pdf["forecast_demand"].astype(float)))
    problem = pulp.LpProblem(f"item_{item_id}_allocation", pulp.LpMinimize)
    allocation = problem.add_variable_dicts(
        "allocation", store_ids, lowBound=0, cat=pulp.LpInteger
    )
    unmet = problem.add_variable_dicts("unmet", store_ids, lowBound=0)
    problem += pulp.lpSum(unmet[store_id] for store_id in store_ids)
    problem += pulp.lpSum(allocation[store_id] for store_id in store_ids) <= total_supply
    for store_id in store_ids:
        problem += allocation[store_id] <= max(0.0, forecast_by_store[store_id])
        problem += unmet[store_id] >= forecast_by_store[store_id] - allocation[store_id]
    solve_stats = problem.solve(cbc_solver)
    if not solve_stats.has_solution or solve_stats.status not in {
        pulp.LpSolveStatus.Optimal,
        pulp.LpSolveStatus.GapLimit,
    }:
        raise RuntimeError(f"Allocation failed for item {item_id}: {solve_stats.status_str}")

    equal_base, equal_remainder = divmod(total_supply, len(store_ids))
    for position, row in item_pdf.iterrows():
        store_id = int(row["store"])
        optimized_allocation = int(round(pulp.value(allocation[store_id])))
        equal_allocation = equal_base + int(position < equal_remainder)
        actual_demand = float(row["actual_demand"])
        allocation_rows.append(
            {
                "item": int(item_id),
                "store": store_id,
                "forecast_demand": float(row["forecast_demand"]),
                "actual_demand": actual_demand,
                "total_item_supply": int(total_supply),
                "optimized_allocation": optimized_allocation,
                "equal_split_allocation": equal_allocation,
                "optimized_actual_unmet": max(actual_demand - optimized_allocation, 0.0),
                "equal_split_actual_unmet": max(actual_demand - equal_allocation, 0.0),
            }
        )

allocation_pdf = pd.DataFrame(allocation_rows)
summary_pdf = (
    allocation_pdf.groupby("item", as_index=False)
    .agg(
        total_item_supply=("total_item_supply", "first"),
        optimized_actual_unmet=("optimized_actual_unmet", "sum"),
        equal_split_actual_unmet=("equal_split_actual_unmet", "sum"),
    )
)
summary_pdf["item"] = summary_pdf["item"].astype(str)
summary_pdf.loc[len(summary_pdf)] = {
    "item": "ALL",
    "total_item_supply": allocation_pdf.groupby("item")["total_item_supply"].first().sum(),
    "optimized_actual_unmet": allocation_pdf["optimized_actual_unmet"].sum(),
    "equal_split_actual_unmet": allocation_pdf["equal_split_actual_unmet"].sum(),
}

allocation_pdf.to_csv(os.path.join(output_path, "inventory_allocation.csv"), index=False)
summary_pdf.to_csv(os.path.join(output_path, "inventory_summary.csv"), index=False)
inventory_pdf.to_csv(os.path.join(output_path, "reorder_points.csv"), index=False)
print(summary_pdf.to_string(index=False))
