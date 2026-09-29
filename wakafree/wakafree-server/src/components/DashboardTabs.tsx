'use client'

function Icon({ d, points }: { d?: string[]; points?: string[] }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {d?.map((p, i) => <path key={i} d={p} />)}
      {points?.map((p, i) => <polyline key={i} points={p} />)}
    </svg>
  )
}

const TABS = [
  {
    href: '/dashboard#timeline',
    label: 'Timeline',
    icon: <Icon points={['22 12 18 12 15 21 9 3 6 12 2 12']} />,
  },
  {
    href: '/dashboard#ai-metrics',
    label: 'Metrics',
    icon: <Icon d={['M3 17l6-6 4 4 8-8', 'M17 7h4v4']} />,
  },
  {
    href: '/dashboard#statistics',
    label: 'Statistics',
    icon: <Icon d={['M12 20V10', 'M18 20V4', 'M6 20v-4']} />,
  },
]

export default function DashboardTabs() {
  return (
    <div className="hidden items-center gap-1 md:flex">
      {TABS.map((tab) => (
        <a
          key={tab.label}
          href={tab.href}
          className="flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-semibold text-onsurface-variant transition-colors hover:bg-container-high hover:text-onsurface"
        >
          {tab.icon}
          {tab.label}
        </a>
      ))}
    </div>
  )
}
