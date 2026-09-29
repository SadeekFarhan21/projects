import DashboardTabs from '@/components/DashboardTabs'

function CodeMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  )
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-surface">
      <nav className="bg-surface/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 py-4">
          <a href="/dashboard" className="flex items-center gap-2.5 text-onsurface">
            <CodeMark />
            <span className="text-3xl font-semibold tracking-tight">Farhan&apos;s WakaTime</span>
          </a>
          <DashboardTabs />
        </div>
      </nav>
      {children}
    </div>
  )
}
