import { formatSeconds, compactNumber } from '@/lib/compute'
import type { ProjectDetail } from '@/lib/projects'

export interface ProjectMeta {
  repository?: string | null
  url?: string | null
  firstHeartbeat?: string | null
  lastHeartbeat?: string | null
}

export default function ProjectDetailCard({
  project,
  meta,
}: {
  project: ProjectDetail
  meta?: ProjectMeta
}) {
  return (
    <div className="bg-container-low border border-line rounded-lg transition-colors hover:border-outline-variant p-5">
      {/* Header */}
      <div className="flex items-center gap-2 mb-1">
        <div className="w-8 h-8 bg-secondary-container/40 rounded-lg flex items-center justify-center shrink-0">
          <span className="text-primary text-xs">📁</span>
        </div>
        <h3 className="font-semibold text-onsurface truncate">{project.name}</h3>
      </div>
      <p className="text-sm text-outline mb-3 pl-10">{formatSeconds(project.seconds)}</p>

      {/* Split AI vs Human bar */}
      <div className="flex h-1.5 rounded-full overflow-hidden mb-5 bg-container-high">
        <div className="bg-primary-dim" style={{ width: `${project.aiPercent}%` }} />
        <div className="bg-[#34d399]" style={{ width: `${project.humanPercent}%` }} />
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-y-5 gap-x-4">
        <div>
          <p className="font-mono text-primary text-[11px] font-medium uppercase tracking-[0.08em] mb-1">⚡ AI Changes</p>
          <p className="text-xl font-bold text-onsurface leading-none">{compactNumber(project.aiChanges)}</p>
          <p className="text-outline text-xs mt-1">{project.aiPercent.toFixed(project.aiPercent < 10 ? 1 : 0)}%</p>
        </div>
        <div>
          <p className="font-mono text-outline text-[11px] font-medium uppercase tracking-[0.08em] mb-1">👤 Human Changes</p>
          <p className="text-xl font-bold text-onsurface leading-none">{compactNumber(project.humanChanges)}</p>
          <p className="text-outline text-xs mt-1">{project.humanPercent.toFixed(project.humanPercent < 10 ? 1 : 0)}%</p>
        </div>

        <div>
          <p className="font-mono text-outline text-[11px] font-medium uppercase tracking-[0.08em] mb-1">💬 AI Prompts</p>
          <p className="text-xl font-bold text-onsurface leading-none">{project.aiPrompts}</p>
        </div>
        <div>
          <p className="font-mono text-outline text-[11px] font-medium uppercase tracking-[0.08em] mb-1">🗨 AI Sessions</p>
          <p className="text-xl font-bold text-onsurface leading-none">{project.aiSessions}</p>
          <p className="text-outline text-xs mt-1">{project.avgPromptsPerSession} avg prompts</p>
        </div>

        <div>
          <p className="font-mono text-outline text-[11px] font-medium uppercase tracking-[0.08em] mb-1">▦ Tokens</p>
          <p className="text-xl font-bold text-onsurface leading-none">
            {compactNumber(project.inputTokens + project.outputTokens)}
          </p>
          <p className="text-outline text-xs mt-1">
            {compactNumber(project.inputTokens)} in / {compactNumber(project.outputTokens)} out
          </p>
        </div>
        <div>
          <p className="font-mono text-outline text-[11px] font-medium uppercase tracking-[0.08em] mb-1">$ AI Spend</p>
          <p className="text-xl font-bold text-onsurface leading-none">${project.aiSpend.toFixed(2)}</p>
          {project.agentBreakdown.length > 0 && (
            <p className="text-outline text-xs mt-1 truncate">
              {project.agentBreakdown
                .map((a) => `${a.name} $${a.cost.toFixed(a.cost < 1 ? 3 : 2)}`)
                .join(', ')}
            </p>
          )}
        </div>
      </div>

      {/* Optional metadata footer */}
      {meta && (meta.lastHeartbeat || meta.repository) && (
        <div className="mt-5 pt-4 border-t border-line space-y-1">
          {meta.lastHeartbeat && (
            <p className="text-xs text-outline">
              Last active: <span className="text-onsurface">{meta.lastHeartbeat}</span>
            </p>
          )}
          {meta.firstHeartbeat && (
            <p className="text-xs text-outline">
              First seen: <span className="text-onsurface">{meta.firstHeartbeat}</span>
            </p>
          )}
          {meta.repository && (
            <a
              href={meta.repository}
              className="text-xs text-primary hover:text-primary-dim truncate block"
            >
              {meta.repository}
            </a>
          )}
        </div>
      )}
    </div>
  )
}
