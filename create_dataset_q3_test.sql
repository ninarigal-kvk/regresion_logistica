-- =============== PARÁMETROS (literales) ===============
-- Ventana Q3 (también cubre lookback de 90d previo a 2025-10-01)
-- as_of_date = 2025-10-01
-- lookback_start = 2025-07-03 (dentro de Q3)
-- etiqueta: TX en Q3 (2025-07-01 .. 2025-10-01)
-- ======================================================

-- Limpieza previa (no falla si no existe)
DROP TABLE IF EXISTS dsq3_sales_qleads;
DROP TABLE IF EXISTS dsq3_sales_bookings;
DROP TABLE IF EXISTS dsq3_sales_tx;
DROP TABLE IF EXISTS dsq3_supply_regs;
DROP TABLE IF EXISTS dsq3_supply_os;
DROP TABLE IF EXISTS dsq3_supply_tx;

-- 1) BASES PREFILTRADAS A Q3 (menor volumen desde el inicio)
CREATE TABLE dsq3_sales_qleads
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
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
  AND opp.createddate::date >= DATE '2025-07-01'
  AND opp.createddate::date <  DATE '2025-10-01';

CREATE TABLE dsq3_sales_bookings
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT DISTINCT
  account__c::varchar     AS user_id,
  opportunity__c::varchar AS opportunity_id,
  paymentdate__c::date    AS event_date
FROM salesforce_ssa_refined.reservation__c
WHERE status__c = 'Capturada'
  AND country__c = 'AR'
  AND account__c IS NOT NULL
  AND opportunity__c IS NOT NULL
  AND paymentdate__c::date >= DATE '2025-07-01'
  AND paymentdate__c::date <  DATE '2025-10-01';

CREATE TABLE dsq3_supply_regs
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT DISTINCT
  o.accountid::varchar          AS user_id,
  regs.opportunity_id::varchar  AS opportunity_id,
  regs.fecha_registro::date     AS event_date,
  regs.km::bigint               AS km      -- <--- cambia si el campo se llama distinto
FROM playground.customer_supply_register_funnel regs
JOIN salesforce_ssa_refined.opportunity o
  ON o.id = regs.opportunity_id
WHERE regs.country_iso = 'AR'
  AND regs.fecha_registro::date >= DATE '2025-07-01'
  AND regs.fecha_registro::date <  DATE '2025-10-01';

CREATE TABLE dsq3_supply_os
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
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
  AND COALESCE(os.fecha_inspeccion_agendada::date, os.fecha_registro::date) >= DATE '2025-07-01'
  AND COALESCE(os.fecha_inspeccion_agendada::date, os.fecha_registro::date) <  DATE '2025-10-01';

-- TX en Q3 (para etiqueta)
CREATE TABLE dsq3_sales_tx
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
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
  AND ofh.createddate::date >= DATE '2025-07-01'
  AND ofh.createddate::date <  DATE '2025-10-01';

CREATE TABLE dsq3_supply_tx
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
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
  AND ofh.createddate::date >= DATE '2025-07-01'
  AND ofh.createddate::date <  DATE '2025-10-01';

-- 2) UNIVERSO (señales en Q3)
DROP TABLE IF EXISTS dsq3_universe;
CREATE TABLE dsq3_universe
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT DISTINCT user_id FROM dsq3_sales_qleads
UNION
SELECT DISTINCT user_id FROM dsq3_sales_bookings
UNION
SELECT DISTINCT user_id FROM dsq3_supply_regs
UNION
SELECT DISTINCT user_id FROM dsq3_supply_os;

-- 3) AGREGADOS DEL LOOKBACK (90 días hasta as_of=2025-10-01)
-- Como ya prefiltramos a Q3 (>=2025-07-01 y <2025-10-01),
-- esto ya coincide con el lookback (arranca 2025-07-03).
DROP TABLE IF EXISTS dsq3_agg_qleads;
CREATE TABLE dsq3_agg_qleads
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT user_id,
       COUNT(*) AS n_qleads_90d,
       MAX(event_date) AS last_qlead
FROM dsq3_sales_qleads
GROUP BY 1;

DROP TABLE IF EXISTS dsq3_agg_bookings;
CREATE TABLE dsq3_agg_bookings
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT user_id,
       COUNT(*) AS n_bookings_90d,
       MAX(event_date) AS last_booking
FROM dsq3_sales_bookings
GROUP BY 1;

DROP TABLE IF EXISTS dsq3_agg_regs;
CREATE TABLE dsq3_agg_regs
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT user_id,
       COUNT(*) AS n_registers_90d,
       MAX(event_date) AS last_register
FROM dsq3_supply_regs
GROUP BY 1;

DROP TABLE IF EXISTS dsq3_agg_os;
CREATE TABLE dsq3_agg_os
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT user_id,
       COUNT(*) AS n_online_schedules_90d,
       MAX(event_date) AS last_os
FROM dsq3_supply_os
GROUP BY 1;

-- 4) KM (último km del lookback)
DROP TABLE IF EXISTS dsq3_last_km;
CREATE TABLE dsq3_last_km
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT user_id, km, event_date,
       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_date DESC) AS rn
FROM dsq3_supply_regs
WHERE km IS NOT NULL;

DROP TABLE IF EXISTS dsq3_km_features;
CREATE TABLE dsq3_km_features
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT
  u.user_id,
  lk.km AS reg_km_last,
  CASE WHEN lk.km IS NOT NULL THEN LOG( (lk.km + 1)::double precision ) END AS reg_km_log1p,
  CASE WHEN lk.km IS NOT NULL THEN LENGTH(CAST(lk.km AS varchar)) - LENGTH(RTRIM(CAST(lk.km AS varchar), '0')) END AS reg_km_trailing_zeros,
  CASE WHEN lk.km IS NOT NULL AND MOD(lk.km,1000)=0 THEN 1 ELSE 0 END AS reg_km_is_round_1000,
  CASE WHEN lk.km IS NOT NULL AND MOD(lk.km, 500)=0 THEN 1 ELSE 0 END AS reg_km_is_round_500,
  CASE WHEN lk.km IS NOT NULL AND MOD(lk.km, 100)=0 THEN 1 ELSE 0 END AS reg_km_is_round_100,
  CASE WHEN lk.km IS NOT NULL AND MOD(lk.km,  50)=0 THEN 1 ELSE 0 END AS reg_km_is_round_50,
  CASE WHEN lk.km IS NOT NULL AND MOD(lk.km,  10)=0 THEN 1 ELSE 0 END AS reg_km_is_round_10,
  CASE WHEN lk.km IS NOT NULL THEN (MOD(lk.km,1000)::double precision)/1000.0 END AS reg_km_mod1000_norm,
  CASE WHEN lk.km IS NOT NULL THEN (MOD(lk.km, 100)::double precision)/100.0  END AS reg_km_mod100_norm
FROM dsq3_universe u
LEFT JOIN (SELECT user_id, km FROM dsq3_last_km WHERE rn=1) lk
       ON lk.user_id = u.user_id;

-- 5) FEATURES (recencias vs as_of = 2025-10-01)
DROP TABLE IF EXISTS dsq3_features;
CREATE TABLE dsq3_features
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT
  u.user_id,
  COALESCE(q.n_qleads_90d,0)           AS n_qleads_90d,
  COALESCE(b.n_bookings_90d,0)         AS n_bookings_90d,
  COALESCE(r.n_registers_90d,0)        AS n_registers_90d,
  COALESCE(o.n_online_schedules_90d,0) AS n_online_schedules_90d,
  DATEDIFF('day', q.last_qlead,    DATE '2025-10-01') AS recency_last_qlead,
  DATEDIFF('day', b.last_booking,  DATE '2025-10-01') AS recency_last_booking,
  DATEDIFF('day', r.last_register, DATE '2025-10-01') AS recency_last_register,
  DATEDIFF('day', o.last_os,       DATE '2025-10-01') AS recency_last_online_schedule
FROM dsq3_universe u
LEFT JOIN dsq3_agg_qleads   q ON q.user_id = u.user_id
LEFT JOIN dsq3_agg_bookings b ON b.user_id = u.user_id
LEFT JOIN dsq3_agg_regs     r ON r.user_id = u.user_id
LEFT JOIN dsq3_agg_os       o ON o.user_id = u.user_id;

-- 6) LABEL (TX en Q3)
DROP TABLE IF EXISTS dsq3_tx_q3;
CREATE TABLE dsq3_tx_q3
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT DISTINCT user_id
FROM (
  SELECT user_id, event_date FROM dsq3_sales_tx
  UNION ALL
  SELECT user_id, event_date FROM dsq3_supply_tx
) t;

-- 7) DATASET FINAL (persistente)
DROP TABLE IF EXISTS dataset_q3_holdout;
CREATE TABLE dataset_q3_holdout
DISTSTYLE KEY DISTKEY(user_id) COMPOUND SORTKEY(user_id) AS
SELECT
  'holdout'::varchar               AS set,
  DATE '2025-10-01'                AS cutoff_date,
  f.user_id,
  f.n_qleads_90d,
  f.n_bookings_90d,
  f.recency_last_qlead,
  f.recency_last_booking,
  f.n_registers_90d,
  f.n_online_schedules_90d,
  f.recency_last_register,
  f.recency_last_online_schedule,
  (f.n_qleads_90d + f.n_registers_90d) AS total_signals_90d,
  k.reg_km_last,
  k.reg_km_log1p,
  k.reg_km_trailing_zeros,
  k.reg_km_is_round_1000,
  k.reg_km_is_round_500,
  k.reg_km_is_round_100,
  k.reg_km_is_round_50,
  k.reg_km_is_round_10,
  k.reg_km_mod1000_norm,
  k.reg_km_mod100_norm,
  CASE WHEN tx.user_id IS NOT NULL THEN 1 ELSE 0 END AS target_30d
FROM dsq3_features f
LEFT JOIN dsq3_km_features k ON k.user_id = f.user_id
LEFT JOIN dsq3_tx_q3 tx       ON tx.user_id = f.user_id
ORDER BY f.user_id;

-- 8) Chequeo
SELECT COUNT(*) AS n_rows,
       SUM(target_30d) AS positives,
       AVG(target_30d::float) AS positive_rate
FROM dataset_q3_holdout;

-- Para testear: python .\training.py --data dataset_q1_q2_kms.csv --outdir outputs_q1q2_km --score_data dataset_q3_holdout.csv --model .\outputs_q1q2_km\model.joblib --score_out scores_q3.csv

