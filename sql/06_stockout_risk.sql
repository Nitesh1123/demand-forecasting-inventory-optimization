-- Store-item pairs with the lowest fill rate under the optimised allocation: the first candidates for extra supply.
SELECT
    store,
    item,
    ROUND(actual_demand, 0) AS actual_demand,
    pulp_allocation,
    ROUND(pulp_actual_unmet, 0) AS unmet_units,
    ROUND(1 - pulp_actual_unmet / actual_demand, 4) AS fill_rate,
    ROUND(safety_stock, 1) AS safety_stock,
    ROUND(reorder_point, 1) AS reorder_point
FROM fact_inventory
ORDER BY fill_rate ASC, unmet_units DESC
LIMIT 20;
