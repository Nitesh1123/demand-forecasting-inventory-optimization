-- Weekly total actual vs. forecast across all stores and items in the forecast window.
SELECT
    MIN(f.date) AS week_start,
    SUM(f.actual) AS actual_units,
    ROUND(SUM(f.forecast), 0) AS forecast_units,
    ROUND(SUM(f.baseline_forecast), 0) AS baseline_units,
    ROUND(100.0 * (SUM(f.forecast) - SUM(f.actual)) / SUM(f.actual), 2) AS forecast_bias_pct
FROM fact_forecast AS f
GROUP BY strftime('%Y-%W', f.date)
ORDER BY week_start;
