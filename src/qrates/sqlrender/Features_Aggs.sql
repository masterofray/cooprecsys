WITH RData AS (
    SELECT
        "{{ user_col }}" AS uid,
        "{{ item_col }}" AS iid,
        COUNT(*) AS freq,
        SUM(CAST("{{ quantity_col }}" AS DOUBLE)) AS total_qty
        {% if total_price_col %}
        ,SUM(CAST("{{ total_price_col }}" AS DOUBLE)) AS total_spend
        {% endif %}
        {% if discount_col %}
        ,1.0 - AVG(CAST("{{ discount_col }}" AS DOUBLE)) AS loyalty_raw
        {% endif %}
        {% if date_col %}
        ,MAX(CAST("{{ date_col }}" AS DATE)) AS last_date
        {% endif %}
    FROM
        FirstData
    GROUP BY
        "{{ user_col }}", "{{ item_col }}"
)

{% if date_col %}
,Recency_Feature AS (
    SELECT *,
        DATEDIFF('day', last_date, MAX(last_date) OVER ()) AS days_ago,
        DATEDIFF('day', MIN(last_date) OVER (), MAX(last_date) OVER ()) AS max_days_ago
    FROM RData
)
{% endif %}

,signals AS (
    SELECT
        uid, iid,
        LN(1.0 + CAST(freq AS DOUBLE)) AS log_freq,
        LN(1.0 + total_qty) AS log_qty
        {% if total_price_col %}
        ,LN(1.0 + total_spend) AS log_spend
        {% endif %}
        {% if discount_col %}
        ,GREATEST(0.0, LEAST(1.0, loyalty_raw)) AS loyalty
        {% endif %}
        {% if date_col %}
        ,CASE WHEN max_days_ago = 0 THEN 1.0
              ELSE 1.0 - CAST(days_ago AS DOUBLE)
                       / CAST(max_days_ago AS DOUBLE)
        END AS recency
        {% endif %}
    FROM {% if date_col %}Recency_Feature{% else %}RData{% endif %}
)

,normalized AS (
    SELECT
        uid, iid
        ,(log_freq - MIN(log_freq) OVER ())
            / NULLIF(MAX(log_freq) OVER () - MIN(log_freq) OVER (), 0.0) AS n_freq
        ,(log_qty - MIN(log_qty) OVER ())
            / NULLIF(MAX(log_qty) OVER () - MIN(log_qty) OVER (), 0.0) AS n_qty
        {% if total_price_col %}
        ,(log_spend - MIN(log_spend) OVER ())
            / NULLIF(MAX(log_spend) OVER () - MIN(log_spend) OVER (), 0.0) AS n_spend
        {% endif %}
        {% if discount_col %}
        ,loyalty AS n_loyalty
        {% endif %}
        {% if date_col %}
        ,recency AS n_recency
        {% endif %}
    FROM signals
)

SELECT * FROM normalized;