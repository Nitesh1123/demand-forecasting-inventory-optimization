-- Unmet demand and fill rate for each allocation strategy, by ABC class.
SELECT
    i.abc_class,
    SUM(v.actual_demand) AS actual_demand,
    SUM(v.pulp_actual_unmet) AS pulp_unmet,
    SUM(v.proportional_actual_unmet) AS proportional_unmet,
    SUM(v.equal_split_actual_unmet) AS equal_split_unmet,
    ROUND(1 - SUM(v.pulp_actual_unmet) / SUM(v.actual_demand), 4) AS pulp_fill_rate,
    ROUND(1 - SUM(v.proportional_actual_unmet) / SUM(v.actual_demand), 4) AS proportional_fill_rate,
    ROUND(1 - SUM(v.equal_split_actual_unmet) / SUM(v.actual_demand), 4) AS equal_split_fill_rate
FROM fact_inventory AS v
JOIN dim_item AS i ON i.item = v.item
GROUP BY i.abc_class
ORDER BY i.abc_class;
