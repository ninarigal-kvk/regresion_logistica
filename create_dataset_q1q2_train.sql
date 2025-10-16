-- ===========================================
-- Dataset para propensión a transaccionar
-- TRAIN:
--   - Universo: señales en Q1 [2025-01-01, 2025-04-01)
--   - Label:    TX en Q1+Q2  [2025-01-01, 2025-07-01)
-- TEST:
--   - Universo: señales en Q2 [2025-04-01, 2025-07-01)
--   - Label:    TX en Q2      [2025-04-01, 2025-07-01)
--
-- Notas:
-- - Recencias: días desde el último evento hasta el fin del período (period_end_excl).
-- - Se mantienen nombres de columnas como *_90d para compatibilidad con tu pipeline.
-- - Incluye features de km (opcional).
-- ===========================================

WITH
-- =========================
-- 0) Parámetros
-- =========================
params AS (
  SELECT
    'train'::varchar AS set,
    '2025-01-01'::date AS period_start,
    '2025-04-01'::date AS period_end_excl,      -- Q1 (end-exclusive)
    '2025-01-01'::date AS label_start,
    '2025-07-01'::date AS label_end_excl,       -- Q1+Q2 (end-exclusive)
    '2025-04-01'::date AS cutoff_date           -- para tu script: fin de Q1 (siguiente día)
  UNION ALL
  SELECT
    'test',
    '2025-04-01'::date,                         -- Q2 start
    '2025-07-01'::date,                         -- Q2 end-exclusive
    '2025-04-01'::date,                         -- label en Q2
    '2025-07-01'::date,
    '2025-07-01'::date                          -- para tu script: fin de Q2 (siguiente día)
),

-- =========================
-- 1) Fuentes base
-- =========================
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
    o.accountid::varchar          AS user_id,
    regs.opportunity_id::varchar  AS opportunity_id,
    regs.fecha_registro::date     AS event_date
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

-- (Opcional) km declarado en supply register, sanitizado
supply_regs_km AS (
  SELECT DISTINCT
    o.accountid::varchar          AS user_id,
    regs.opportunity_id::varchar  AS opportunity_id,
    regs.fecha_registro::date     AS event_date,
    NULLIF(REGEXP_REPLACE(regs.km::varchar, '[^0-9]', ''), '')::bigint AS km_num
  FROM playground.customer_supply_register_funnel regs
  JOIN salesforce_ssa_refined.opportunity o
    ON o.id = regs.opportunity_id
  WHERE regs.country_iso = 'AR'
),

-- =========================
-- 2) Universo por set (período)
-- =========================
universe AS (
  SELECT DISTINCT
    p.set,
    p.cutoff_date,
    p.period_start,
    p.period_end_excl,
    p.label_start,
    p.label_end_excl,
    u.user_id
  FROM params p
  JOIN (
    SELECT user_id, event_date FROM sales_qleads
    UNION ALL SELECT user_id, event_date FROM sales_bookings
    UNION ALL SELECT user_id, event_date FROM supply_regs
    UNION ALL SELECT user_id, event_date FROM supply_online_sched
  ) u
    ON u.event_date >= p.period_start
   AND u.event_date <  p.period_end_excl
),

-- =========================
-- 3) Features dentro del período
-- =========================
feat_sales AS (
  SELECT
    u.set,
    u.cutoff_date,
    u.period_start,
    u.period_end_excl,
    u.user_id,
    /* Conteos en el período */
    (SELECT COUNT(*) FROM sales_qleads q
      WHERE q.user_id = u.user_id
        AND q.event_date >= u.period_start
        AND q.event_date <  u.period_end_excl
    ) AS n_qleads_90d,
    (SELECT COUNT(*) FROM sales_bookings b
      WHERE b.user_id = u.user_id
        AND b.event_date >= u.period_start
        AND b.event_date <  u.period_end_excl
    ) AS n_bookings_90d,
    /* Recencias al fin del período */
    DATEDIFF('day',
      (SELECT MAX(q.event_date) FROM sales_qleads q
        WHERE q.user_id = u.user_id
          AND q.event_date >= u.period_start
          AND q.event_date <  u.period_end_excl),
      u.period_end_excl
    ) AS recency_last_qlead,
    DATEDIFF('day',
      (SELECT MAX(b.event_date) FROM sales_bookings b
        WHERE b.user_id = u.user_id
          AND b.event_date >= u.period_start
          AND b.event_date <  u.period_end_excl),
      u.period_end_excl
    ) AS recency_last_booking
  FROM universe u
),
feat_supply AS (
  SELECT
    u.set,
    u.cutoff_date,
    u.period_start,
    u.period_end_excl,
    u.user_id,
    (SELECT COUNT(*) FROM supply_regs r
      WHERE r.user_id = u.user_id
        AND r.event_date >= u.period_start
        AND r.event_date <  u.period_end_excl
    ) AS n_registers_90d,
    (SELECT COUNT(*) FROM supply_online_sched s
      WHERE s.user_id = u.user_id
        AND s.event_date >= u.period_start
        AND s.event_date <  u.period_end_excl
    ) AS n_online_schedules_90d,
    DATEDIFF('day',
      (SELECT MAX(r.event_date) FROM supply_regs r
        WHERE r.user_id = u.user_id
          AND r.event_date >= u.period_start
          AND r.event_date <  u.period_end_excl),
      u.period_end_excl
    ) AS recency_last_register,
    DATEDIFF('day',
      (SELECT MAX(s.event_date) FROM supply_online_sched s
        WHERE s.user_id = u.user_id
          AND s.event_date >= u.period_start
          AND s.event_date <  u.period_end_excl),
      u.period_end_excl
    ) AS recency_last_online_schedule,

    -- Último km declarado dentro del período
    (
      SELECT rk.km_num
      FROM supply_regs_km rk
      WHERE rk.user_id = u.user_id
        AND rk.event_date >= u.period_start
        AND rk.event_date <  u.period_end_excl
      ORDER BY rk.event_date DESC
      LIMIT 1
    ) AS reg_km_last
  FROM universe u
),

-- Derivados de km (redondez)
feat_km AS (
  SELECT
    set,
    cutoff_date,
    user_id,
    reg_km_last,
    CASE
      WHEN reg_km_last IS NULL THEN NULL
      ELSE LEN(reg_km_last::varchar) - LEN(REGEXP_REPLACE(reg_km_last::varchar, '0+$', ''))
    END AS reg_km_trailing_zeros,
    CASE WHEN reg_km_last IS NOT NULL AND reg_km_last % 1000 = 0 THEN 1 ELSE 0 END AS reg_km_is_round_1000,
    CASE WHEN reg_km_last IS NOT NULL AND reg_km_last %  500 = 0 THEN 1 ELSE 0 END AS reg_km_is_round_500,
    CASE WHEN reg_km_last IS NOT NULL AND reg_km_last %  100 = 0 THEN 1 ELSE 0 END AS reg_km_is_round_100,
    CASE WHEN reg_km_last IS NOT NULL AND reg_km_last %   50 = 0 THEN 1 ELSE 0 END AS reg_km_is_round_50,
    CASE WHEN reg_km_last IS NOT NULL AND reg_km_last %   10 = 0 THEN 1 ELSE 0 END AS reg_km_is_round_10,
    CASE
      WHEN reg_km_last IS NULL THEN NULL
      ELSE LEAST(reg_km_last % 1000, 1000 - (reg_km_last % 1000))::double precision / 1000.0
    END AS reg_km_mod1000_norm,
    CASE
      WHEN reg_km_last IS NULL THEN NULL
      ELSE LEAST(reg_km_last % 100, 100 - (reg_km_last % 100))::double precision / 100.0
    END AS reg_km_mod100_norm,
    CASE
      WHEN reg_km_last IS NULL THEN NULL
      ELSE LOG( (reg_km_last::double precision) + 1.0 )
    END AS reg_km_log1p
  FROM feat_supply
),

-- =========================
-- 4) Labels por set (período de evaluación)
-- =========================
labels AS (
  SELECT
    u.set,
    u.cutoff_date,
    u.user_id,
    CASE WHEN EXISTS (
      SELECT 1
      FROM (
        SELECT user_id, event_date FROM sales_tx
        UNION ALL
        SELECT user_id, event_date FROM supply_tx
      ) tx
      WHERE tx.user_id = u.user_id
        AND tx.event_date >= u.label_start
        AND tx.event_date <  u.label_end_excl
    ) THEN 1 ELSE 0 END AS target_30d
  FROM universe u
),

-- =========================
-- 5) Dataset final
-- =========================
dataset AS (
  SELECT
    u.set,
    u.cutoff_date::date,
    u.user_id,

    -- Sales
    COALESCE(fs.n_qleads_90d,0)   AS n_qleads_90d,
    COALESCE(fs.n_bookings_90d,0) AS n_bookings_90d,
    fs.recency_last_qlead,
    fs.recency_last_booking,

    -- Supply
    COALESCE(fp.n_registers_90d,0)        AS n_registers_90d,
    COALESCE(fp.n_online_schedules_90d,0) AS n_online_schedules_90d,
    fp.recency_last_register,
    fp.recency_last_online_schedule,

    -- Totales (mantenemos tu definición)
    COALESCE(fs.n_qleads_90d,0)
      + COALESCE(fp.n_registers_90d,0)    AS total_signals_90d,

    -- KM features (opcionales)
    km.reg_km_last,
    km.reg_km_log1p,
    km.reg_km_trailing_zeros,
    km.reg_km_is_round_1000,
    km.reg_km_is_round_500,
    km.reg_km_is_round_100,
    km.reg_km_is_round_50,
    km.reg_km_is_round_10,
    km.reg_km_mod1000_norm,
    km.reg_km_mod100_norm,

    -- Label
    l.target_30d
  FROM universe u
  LEFT JOIN feat_sales  fs ON fs.set = u.set AND fs.cutoff_date = u.cutoff_date AND fs.user_id = u.user_id
  LEFT JOIN feat_supply fp ON fp.set = u.set AND fp.cutoff_date = u.cutoff_date AND fp.user_id = u.user_id
  LEFT JOIN feat_km     km ON km.set = u.set AND km.cutoff_date = u.cutoff_date AND km.user_id = u.user_id
  LEFT JOIN labels      l  ON l.set = u.set AND l.cutoff_date = u.cutoff_date AND l.user_id = u.user_id
)

SELECT *
FROM dataset
ORDER BY set, cutoff_date, user_id;

-- Para entrenar: python .\training.py --data dataset_q1_q2_kms.csv --outdir outputs_q1q2_km --topk 0.10 --calibrate isotonic --penalty l2 --C 1.0
