-- ============================================================
-- HOUSING AFFORDABILITY DASHBOARD — Supabase Schema
-- Run in: Supabase Dashboard → SQL Editor → Run
-- ============================================================

-- 1. USERS (replaces users.json)
CREATE TABLE IF NOT EXISTS users (
    id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    username      TEXT        UNIQUE NOT NULL,
    full_name     TEXT        NOT NULL DEFAULT '',
    email         TEXT        NOT NULL DEFAULT '',
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- 2. CLUSTER STATS (pre-computed aggregates, replaces CSV groupby on startup)
CREATE TABLE IF NOT EXISTS cluster_stats (
    cluster_id          INTEGER PRIMARY KEY,
    zinc2               FLOAT,   -- avg annual income
    costmed             FLOAT,   -- avg monthly housing cost
    cost_burden_ratio   FLOAT,   -- avg burden ratio
    age1                FLOAT,   -- avg householder age
    bedrms              FLOAT,   -- avg bedrooms
    count               INTEGER  -- total households in cluster
);

-- 3. SCATTER DATA (2 000-row sample for the Income vs Burden chart)
CREATE TABLE IF NOT EXISTS scatter_data (
    id                  SERIAL PRIMARY KEY,
    zinc2               FLOAT,
    cost_burden_ratio   FLOAT,
    cluster_label       INTEGER,
    costmed             FLOAT
);

-- ── Row-Level Security ────────────────────────────────────────
ALTER TABLE users        ENABLE ROW LEVEL SECURITY;
ALTER TABLE cluster_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE scatter_data  ENABLE ROW LEVEL SECURITY;

-- Service-role key (used by the Python backend) gets full access
CREATE POLICY "service_role_users"        ON users        FOR ALL USING (true);
CREATE POLICY "service_role_cluster_stats" ON cluster_stats FOR ALL USING (true);
CREATE POLICY "service_role_scatter_data"  ON scatter_data  FOR ALL USING (true);

-- ── Storage bucket for .pkl model files ───────────────────────
-- Run this separately in the Storage section OR via SQL:
INSERT INTO storage.buckets (id, name, public)
VALUES ('models', 'models', false)
ON CONFLICT DO NOTHING;
