import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import clsx from 'clsx';
import { Maximize2, X } from 'lucide-react';

interface TileProps {
  title: string;
  icon?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
  accentColor?: string;
  badge?: React.ReactNode;
  /** Set false to hide the expand button (e.g. on StatsOverview) */
  expandable?: boolean;
}

export function Tile({ title, icon, className, children, accentColor, badge, expandable = true }: TileProps) {
  const [expanded, setExpanded] = useState(false);

  const close = useCallback(() => setExpanded(false), []);

  // Escape key to close
  useEffect(() => {
    if (!expanded) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [expanded, close]);

  // Lock body scroll when expanded
  useEffect(() => {
    if (expanded) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.body.style.overflow = ''; };
  }, [expanded]);

  const headerContent = (
    <>
      <div className="flex items-center gap-2">
        {icon && <span className="text-lg">{icon}</span>}
        <h3
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: accentColor || '#9ca3af' }}
        >
          {title}
        </h3>
      </div>
      <div className="flex items-center gap-2">
        {badge}
        {expandable && !expanded && (
          <button
            onClick={(e) => { e.stopPropagation(); setExpanded(true); }}
            className="p-1 rounded-lg text-gray-500 hover:text-white/70 hover:bg-white/5 transition-all opacity-0 group-hover:opacity-100"
            title="Expand"
          >
            <Maximize2 size={13} />
          </button>
        )}
      </div>
    </>
  );

  /* ── Inline (grid) tile ── */
  const inlineTile = (
    <div
      className={clsx(
        'group relative bg-gray-900/30 backdrop-blur-2xl border border-gray-700/15 rounded-2xl p-4 overflow-hidden glass-tile',
        'hover:border-gray-600/30 transition-all duration-300',
        'shadow-lg shadow-black/20',
        className
      )}
      style={{
        background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.5) 0%, rgba(10, 10, 18, 0.4) 100%)',
      }}
    >
      {/* Subtle top accent line */}
      <div
        className="absolute top-0 left-4 right-4 h-px opacity-30"
        style={{ background: `linear-gradient(90deg, transparent, ${accentColor || '#6b7280'}, transparent)` }}
      />

      <div className="flex items-center justify-between mb-3">
        {headerContent}
      </div>
      <div className="h-[calc(100%-2.5rem)]">{children}</div>
    </div>
  );

  if (!expanded) return inlineTile;

  /* ── Expanded full-screen modal (portal) ── */
  const modal = (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-6 expanded-tile-backdrop"
      onClick={(e) => { if (e.target === e.currentTarget) close(); }}
    >
      {/* Backdrop with heavy blur */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-xl" />

      {/* Liquid glass container */}
      <div
        className="relative w-full h-full max-w-[75vw] max-h-[75vh] rounded-3xl overflow-hidden expanded-tile-glass"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.85) 0%, rgba(10, 10, 18, 0.8) 50%, rgba(15, 15, 30, 0.85) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: `
            0 0 0 1px rgba(255, 255, 255, 0.03),
            0 0 80px -20px rgba(255, 255, 255, 0.08),
            0 32px 80px -20px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.04)
          `,
          backdropFilter: 'blur(40px) saturate(1.2)',
        }}
      >
        {/* Top accent line */}
        <div
          className="absolute top-0 left-8 right-8 h-px opacity-40"
          style={{ background: `linear-gradient(90deg, transparent, ${accentColor || '#ffffff'}, transparent)` }}
        />

        {/* Refraction highlight — liquid glass effect */}
        <div
          className="absolute top-0 left-0 right-0 h-32 pointer-events-none"
          style={{
            background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, transparent 100%)',
          }}
        />

        {/* Header */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-gray-700/20">
          <div className="flex items-center gap-3">
            {icon && <span className="text-xl">{icon}</span>}
            <h2
              className="text-sm font-bold uppercase tracking-wider"
              style={{ color: accentColor || '#ffffff' }}
            >
              {title}
            </h2>
            {badge && <div className="ml-2">{badge}</div>}
          </div>
          <button
            onClick={close}
            className="p-2 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-all"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content area — full remaining height */}
        <div className="h-[calc(100%-4.5rem)] p-6 overflow-y-auto scrollbar-thin">
          {children}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {inlineTile}
      {createPortal(modal, document.body)}
    </>
  );
}
