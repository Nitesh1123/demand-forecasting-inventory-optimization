-- Year-over-year sales growth per store.
WITH yearly AS (
    SELECT store, CAST(strftime('%Y', date) AS INTEGER) AS year, SUM(sales) AS total_sales
    FROM fact_sales
    GROUP BY store, year
)
SELECT
    store,
    year,
    total_sales,
    LAG(total_sales) OVER (PARTITION BY store ORDER BY year) AS previous_year_sales,
    ROUND(
        100.0 * (total_sales - LAG(total_sales) OVER (PARTITION BY store ORDER BY year))
        / LAG(total_sales) OVER (PARTITION BY store ORDER BY year),
        2
    ) AS yoy_growth_pct
FROM yearly
ORDER BY store, year;
