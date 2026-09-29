// Shared mapping from a waka_daily row to the timeline payload used by the
// dashboard cards and the /api/timeline route.

export interface TimelineBlock {
  time: number
  duration: number
  project: string
}

export interface TimelinePayload {
  date: string
  totalText: string | null
  projectBlocks: TimelineBlock[]
  slices: Record<string, TimelineBlock[]>
}

interface DurationsWrapper {
  data?: Array<Record<string, unknown> & { time: number; duration: number }>
}

export interface TimelineRow {
  data?: { grand_total?: { text?: string } } | null
  durations?: DurationsWrapper | null
  durations_category?: DurationsWrapper | null
  durations_slices?: Record<string, DurationsWrapper | null> | null
}

// Canonical display names for machines that have multiple WakaTime hostnames.
// Single source of truth — the dashboard imports this too.
export const MACHINE_ALIASES: Record<string, string> = {
  // 'raw-hostname.local': 'Friendly Name',
}

export function normalizeMachineName(name: string): string {
  if (MACHINE_ALIASES[name]) return MACHINE_ALIASES[name]
  return name
}

// Each slice's blocks carry their own field name (category, language, ...).
function blockLabel(b: Record<string, unknown>): string {
  const raw = String(
    b.category ??
      b.language ??
      b.editor ??
      b.machine ??
      b.operating_system ??
      b.os ??
      b.project ??
      'Other'
  )
  // Normalize machine names if this block is machine-sliced.
  if (b.machine !== undefined) return normalizeMachineName(raw)
  return raw
}

function toBlocks(wrap: DurationsWrapper | null | undefined): TimelineBlock[] {
  return (wrap?.data ?? []).map((b) => ({
    time: b.time,
    duration: b.duration,
    project: blockLabel(b),
  }))
}

export function mapTimelineRow(date: string, row: TimelineRow | null): TimelinePayload {
  const slices: Record<string, TimelineBlock[]> = {}
  if (row?.durations_slices) {
    for (const [key, wrap] of Object.entries(row.durations_slices)) {
      const blocks = toBlocks(wrap)
      if (blocks.length > 0) slices[key] = blocks
    }
  }
  // Older syncs only have the category slice in its own column.
  if (!slices.category && row?.durations_category?.data?.length) {
    slices.category = toBlocks(row.durations_category)
  }

  return {
    date,
    totalText: row?.data?.grand_total?.text ?? null,
    projectBlocks: toBlocks(row?.durations),
    slices,
  }
}
