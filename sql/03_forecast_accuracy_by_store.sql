-- Forecast accuracy per store in the 90-day forecast window, LightGBM vs. seasonal-naive baseline.
SELECT
    store,
    SUM(actual) AS actual_units,
    ROUND(SUM(forecast), 0) AS forecast_units,
    ROUND(SUM(absolute_error) / SUM(actual), 4) AS lightgbm_wape,
    ROUND(SUM(baseline_absolute_error) / SUM(actual), 4) AS baseline_wape,
    ROUND((SUM(forecast) - SUM(actual)) / SUM(actual), 4) AS lightgbm_bias,
    ROUND(1 - SUM(absolute_error) / SUM(baseline_absolute_error), 4) AS error_reduction_vs_baseline
FROM fact_forecast
GROUP BY store
ORDER BY lightgbm_wape DESC;
