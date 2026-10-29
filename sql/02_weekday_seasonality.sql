-- Average daily sales by weekday, indexed to the overall daily average (100 = average day).
WITH daily AS (
    SELECT s.date, d.day_of_week_number, d.day_of_week_name, SUM(s.sales) AS total_sales
    FROM fact_sales AS s
    JOIN dim_date AS d ON d.date = s.date
    GROUP BY s.date, d.day_of_week_number, d.day_of_week_name
)
SELECT
    day_of_week_name,
    ROUND(AVG(total_sales), 1) AS avg_daily_sales,
    ROUND(100.0 * AVG(total_sales) / (SELECT AVG(total_sales) FROM daily), 1) AS seasonality_index
FROM daily
GROUP BY day_of_week_number, day_of_week_name
ORDER BY day_of_week_number;
