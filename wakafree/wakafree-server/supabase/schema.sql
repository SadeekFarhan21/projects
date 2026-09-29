-- Run this in your Supabase SQL editor to set up the database

CREATE TABLE IF NOT EXISTS users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email TEXT UNIQUE,
  username TEXT UNIQUE,
  display_name TEXT,
  api_key TEXT UNIQUE DEFAULT gen_random_uuid()::TEXT,
  timezone TEXT DEFAULT 'UTC',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  modified_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS heartbeats (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  entity TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'file',
  time DOUBLE PRECISION NOT NULL,
  project TEXT,
  language TEXT,
  branch TEXT,
  editor TEXT,
  operating_system TEXT,
  machine TEXT,
  is_write BOOLEAN DEFAULT FALSE,
  category TEXT DEFAULT 'coding',
  lines INTEGER,
  lineno INTEGER,
  cursorpos INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_user_id ON heartbeats(user_id);
CREATE INDEX IF NOT EXISTS idx_heartbeats_time ON heartbeats(time);
CREATE INDEX IF NOT EXISTS idx_heartbeats_user_time ON heartbeats(user_id, time);
CREATE INDEX IF NOT EXISTS idx_heartbeats_project ON heartbeats(user_id, project);
CREATE INDEX IF NOT EXISTS idx_heartbeats_language ON heartbeats(user_id, language);

-- WakaTime daily archive — one row per day, full API response stored as JSONB
-- Nothing is discarded: projects, languages, editors, machines, OS, categories,
-- dependencies, AI agent costs/breakdowns, cumulative_total, daily_average — all preserved.
CREATE TABLE IF NOT EXISTS waka_daily (
  date DATE PRIMARY KEY,
  data JSONB NOT NULL,
  durations JSONB,           -- /durations (timeline blocks, sliced by project)
  durations_category JSONB,  -- /durations?slice_by=category (AI Coding vs Coding)
  durations_slices JSONB,    -- map of slice_by → /durations (category/language/editor/os/machine)
  synced_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_waka_daily_date ON waka_daily(date DESC);
