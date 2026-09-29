# WakaFree (blog write-up folder)

WakaFree is a WakaTime-compatible coding-time tracker built solo by Farhan Sadeek. A VS Code extension sends heartbeats to a Next.js and Supabase server that speaks WakaTime's `/api/v1` shape, archives upstream daily summaries and duration timelines as JSONB, and renders a dashboard. A FastMCP server exposes WakaTime endpoints as tools. Public repo: SadeekFarhan21/WakaFree, live at https://wakafree.farhansadeek.com.

## Contents

- `wakafree-extension/`: VS Code extension (heartbeat debounce, 30 s batch flush, retry).
- `wakafree-server/`: Next.js app, `/api/v1` routes, WakaTime archive sync, dashboard, Supabase schema.
- `wakafree-mcp/`: FastMCP server for the WakaTime API.
- `results/`: measurements run against this copy (see the command header in each file).

## What was left out of this copy

Environment files and all secret values, CI workflow files that carry a deployment secret, personal machine names and timezone schedule values (replaced by empty or neutral placeholders), and a local copy of WakaTime's developer docs (see https://wakatime.com/developers). A few server routes and helper details were trimmed. Lockfiles are included.
