-- Items ranked by forecast error, with ABC class, so planners know where to review forecasts first.
SELECT
    f.item,
    i.abc_class,
    SUM(f.actual) AS actual_units,
    ROUND(SUM(f.absolute_error) / SUM(f.actual), 4) AS lightgbm_wape,
    ROUND((SUM(f.forecast) - SUM(f.actual)) / SUM(f.actual), 4) AS lightgbm_bias,
    RANK() OVER (ORDER BY SUM(f.absolute_error) / SUM(f.actual) DESC) AS error_rank
FROM fact_forecast AS f
JOIN dim_item AS i ON i.item = f.item
GROUP BY f.item, i.abc_class
ORDER BY error_rank;
