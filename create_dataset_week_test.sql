-- =============================
-- WEEKLY HOLDOUT DATASET (Redshift)
-- Cohorte: cuentas creadas en la semana
-- Features as-of week_end | Horizonte: 30 días
-- =============================

-- ⚠️ Ajustá estas fechas:
WITH params AS (
  SELECT
    'test'::varchar AS set,
    DATE '2025-10-08'::date AS week_start,    -- p.ej. '2025-03-10'
    DATE '2025-10-14'::date   AS week_end       -- p.ej. '2025-03-16'
),
p AS (
  SELECT
    set,
    week_start,
    week_end,
    week_end::date                          AS cutoff_date,
    DATEADD('day',-90, week_end::date)      AS lb_date,
    week_end::date                          AS as_of_date,
    DATEADD('day', 30, week_end::date)      AS horizon_end
  FROM params
),

-- ===== Eventos base (sin filtros de fecha aquí) =====
sales_qleads AS (
  SELECT DISTINCT
    opp.accountid::varchar AS user_id,
    opp.opp_id::varchar    AS opportunity_id,
    opp.createddate::date  AS event_date
  FROM playground.hsa_data_transaction_leads opp
  JOIN playground.hsa_data_salesforce_ssa_refined_account a
    ON a.id = opp.accountid
  WHERE opp.country = 'AR'
    AND opp.origen <> 'Meli wpp'
    AND a.personmobilephone NOT LIKE '%+52%'
),
sales_bookings AS (
  SELECT DISTINCT
    account__c::varchar     AS user_id,
    opportunity__c::varchar AS opportunity_id,
    paymentdate__c::date    AS event_date
  FROM salesforce_ssa_refined.reservation__c
  WHERE status__c = 'Capturada'
    AND country__c = 'AR'
    AND account__c IS NOT NULL
    AND opportunity__c IS NOT NULL
),
sales_tx AS (
  SELECT DISTINCT
    o.accountid::varchar  AS user_id,
    o.id::varchar         AS opportunity_id,
    ofh.createddate::date AS event_date
  FROM salesforce_ssa_refined.opportunityfieldhistory ofh
  JOIN salesforce_ssa_refined.opportunity o
    ON o.id = ofh.opportunityid
  WHERE ofh.field = 'StageName'
    AND ofh.newvalue = 'Cerrada ganada'
    AND o.country__c = 'AR'
    AND o.recordtypename__c IN ('Ventas_ARG','Sales_Global')
),
supply_regs AS (
  SELECT DISTINCT
    o.accountid::varchar         AS user_id,
    regs.opportunity_id::varchar AS opportunity_id,
    regs.fecha_registro::date    AS event_date,
    regs.km                      AS reg_km_last       -- si existiera
  FROM playground.customer_supply_register_funnel regs
  JOIN salesforce_ssa_refined.opportunity o
    ON o.id = regs.opportunity_id
  WHERE regs.country_iso = 'AR'
),
supply_online_sched AS (
  SELECT DISTINCT
    o.accountid::varchar        AS user_id,
    os.opportunity_id::varchar  AS opportunity_id,
    COALESCE(os.fecha_inspeccion_agendada::date, os.fecha_registro::date) AS event_date
  FROM playground.customer_supply_register_funnel os
  JOIN salesforce_ssa_refined.opportunity o
    ON o.id = os.opportunity_id
  WHERE os.country_iso = 'AR'
    AND os.flag_register = TRUE
    AND os.flag_online_schedule = TRUE
),
supply_tx AS (
  SELECT DISTINCT
    o.accountid::varchar  AS user_id,
    o.id::varchar         AS opportunity_id,
    ofh.createddate::date AS event_date
  FROM salesforce_ssa_refined.opportunityfieldhistory ofh
  JOIN salesforce_ssa_refined.opportunity o
    ON o.id = ofh.opportunityid
  WHERE ofh.field = 'StageName'
    AND ofh.newvalue = 'Cerrada ganada'
    AND o.country__c = 'AR'
    AND o.recordtypename__c IN ('Compras','Supply_Global')
),

-- ===== Universo: CUENTAS NUEVAS creadas en la semana =====
accounts_week AS (
  SELECT
    p.set,
    p.cutoff_date,
    a.id::varchar AS user_id
  FROM p
  JOIN playground.hsa_data_salesforce_ssa_refined_account a
    ON a.createddate::date >= p.week_start
   AND a.createddate::date <= p.week_end
  WHERE a.personmobilephone NOT LIKE '%+52%'
),

-- ===== Ventanas de lookback =====
wins AS (
  SELECT
    p.set, p.cutoff_date, p.lb_date, p.as_of_date,
    DATEADD('day',-7,  p.as_of_date) AS lb7,
    DATEADD('day',-30, p.as_of_date) AS lb30
  FROM p
),

-- ===== Conteos 90d (y recencias) por usuario =====
agg_qlead AS (
  SELECT a.user_id,
         COUNT(CASE WHEN q.event_date >= w.lb_date AND q.event_date < w.as_of_date THEN 1 END) AS n_qleads_90d,
         MAX(CASE WHEN q.event_date < w.as_of_date THEN q.event_date END) AS last_qlead
  FROM accounts_week a CROSS JOIN wins w
  LEFT JOIN sales_qleads q ON q.user_id = a.user_id
  GROUP BY a.user_id
),
agg_booking AS (
  SELECT a.user_id,
         COUNT(CASE WHEN b.event_date >= w.lb_date AND b.event_date < w.as_of_date THEN 1 END) AS n_bookings_90d,
         MAX(CASE WHEN b.event_date < w.as_of_date THEN b.event_date END) AS last_booking
  FROM accounts_week a CROSS JOIN wins w
  LEFT JOIN sales_bookings b ON b.user_id = a.user_id
  GROUP BY a.user_id
),
agg_reg AS (
  SELECT a.user_id,
         COUNT(CASE WHEN r.event_date >= w.lb_date AND r.event_date < w.as_of_date THEN 1 END) AS n_registers_90d,
         MAX(CASE WHEN r.event_date < w.as_of_date THEN r.event_date END) AS last_register,
         MAX(CASE WHEN r.event_date < w.as_of_date THEN r.reg_km_last END) AS reg_km_last
  FROM accounts_week a CROSS JOIN wins w
  LEFT JOIN supply_regs r ON r.user_id = a.user_id
  GROUP BY a.user_id
),
agg_os AS (
  SELECT a.user_id,
         COUNT(CASE WHEN s.event_date >= w.lb_date AND s.event_date < w.as_of_date THEN 1 END) AS n_online_schedules_90d,
         MAX(CASE WHEN s.event_date < w.as_of_date THEN s.event_date END) AS last_os
  FROM accounts_week a CROSS JOIN wins w
  LEFT JOIN supply_online_sched s ON s.user_id = a.user_id
  GROUP BY a.user_id
),

-- ===== Ventanas 7d/30d =====
agg_7_30 AS (
  SELECT a.user_id,
         COUNT(CASE WHEN q.event_date >= w.lb7  AND q.event_date < w.as_of_date THEN 1 END) AS n_qleads_7d,
         COUNT(CASE WHEN q.event_date >= w.lb30 AND q.event_date < w.as_of_date THEN 1 END) AS n_qleads_30d,
         COUNT(CASE WHEN r.event_date >= w.lb7  AND r.event_date < w.as_of_date THEN 1 END) AS n_registers_7d,
         COUNT(CASE WHEN r.event_date >= w.lb30 AND r.event_date < w.as_of_date THEN 1 END) AS n_registers_30d
  FROM accounts_week a CROSS JOIN wins w
  LEFT JOIN sales_qleads q ON q.user_id = a.user_id
  LEFT JOIN supply_regs  r ON r.user_id = a.user_id
  GROUP BY a.user_id
),

-- ===== Target en 30 días =====
labels AS (
  SELECT
    p.set,
    p.cutoff_date,
    a.user_id,
    CASE WHEN EXISTS (
      SELECT 1 FROM (
        SELECT user_id, event_date FROM sales_tx
        UNION ALL
        SELECT user_id, event_date FROM supply_tx
      ) tx
      WHERE tx.user_id = a.user_id
        AND tx.event_date >= p.as_of_date
        AND tx.event_date <  p.horizon_end
    ) THEN 1 ELSE 0 END AS target_30d
  FROM p
  JOIN accounts_week a ON 1=1
),

-- ===== Dataset final =====
dataset AS (
  SELECT
    p.set,
    p.cutoff_date,
    a.user_id,
    COALESCE(q.n_qleads_90d,0)             AS n_qleads_90d,
    COALESCE(b.n_bookings_90d,0)           AS n_bookings_90d,
    COALESCE(r.n_registers_90d,0)          AS n_registers_90d,
    COALESCE(os.n_online_schedules_90d,0)  AS n_online_schedules_90d,

    DATEDIFF('day', q.last_qlead,    p.cutoff_date) AS recency_last_qlead,
    DATEDIFF('day', b.last_booking,  p.cutoff_date) AS recency_last_booking,
    DATEDIFF('day', r.last_register, p.cutoff_date) AS recency_last_register,
    DATEDIFF('day', os.last_os,      p.cutoff_date) AS recency_last_online_schedule,

    -- señales totales
    COALESCE(q.n_qleads_90d,0) + COALESCE(r.n_registers_90d,0) AS total_signals_90d,

    -- extras (opcionales para tu modelo FE/KM)
    COALESCE(a73.n_qleads_7d,0)       AS n_qleads_7d,
    COALESCE(a73.n_qleads_30d,0)      AS n_qleads_30d,
    COALESCE(a73.n_registers_7d,0)    AS n_registers_7d,
    COALESCE(a73.n_registers_30d,0)   AS n_registers_30d,
    CASE WHEN COALESCE(r.n_registers_90d,0)+COALESCE(os.n_online_schedules_90d,0)
              >= COALESCE(q.n_qleads_90d,0)
         THEN 'supply' ELSE 'sales' END    AS origin_inferred,
    r.reg_km_last,

    l.target_30d
  FROM p
  JOIN accounts_week a ON 1=1
  LEFT JOIN agg_qlead   q  ON q.user_id = a.user_id
  LEFT JOIN agg_booking b  ON b.user_id = a.user_id
  LEFT JOIN agg_reg     r  ON r.user_id = a.user_id
  LEFT JOIN agg_os      os ON os.user_id = a.user_id
  LEFT JOIN agg_7_30    a73 ON a73.user_id = a.user_id
  LEFT JOIN labels      l  ON l.user_id  = a.user_id AND l.cutoff_date = p.cutoff_date
)

SELECT *
FROM dataset
ORDER BY user_id;

-- descargo como dataset_week_2025-mesinicio-diainicio_2025-mesfin-diafin.csv
-- python .\training.py --data dataset_q1_q2_kms.csv --outdir outputs_week_mesdiainicio_mesdiafin --score_data dataset_week_2025-mesinicio-diainicio_2025-mesfin-diafin.csv --model .\outputs_q1q2_km\model.joblib --score_out scores_week.csv

