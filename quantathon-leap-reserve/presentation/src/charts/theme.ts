// Shared design tokens and formatters for Chart.js charts — Executive Clarity palette
export const COLORS = {
  primary: '#2B52B4',
  primaryDeep: '#1E3A8A',
  primaryLight: '#8BB8F0',
  primaryGlow: 'rgba(43,82,180,0.12)',
  accent: '#E97A2E',
  accentDeep: '#C45A12',
  accentGlow: 'rgba(233,122,46,0.12)',
  ink: '#191C1E',
  inkSoft: '#42474E',
  muted: '#72787E',
  dim: '#ADB3B9',
  surfaceDim: '#F0F2F5',
  surface: '#FFFFFF',
  red: '#DC2626',
  redGlow: 'rgba(220,38,38,0.12)',
  purple: '#7C3AED',
  success: '#16A34A',
} as const;

export const FONT_FAMILY = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";

export function currencyFormatter(value: number | string): string {
  const n = typeof value === 'string' ? parseFloat(value) : value;
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

export function pctFormatter(value: number | string): string {
  const n = typeof value === 'string' ? parseFloat(value) : value;
  return `${(n * 100).toFixed(1)}%`;
}

export const TOOLTIP_DEFAULTS = {
  backgroundColor: 'rgba(25,28,30,0.95)',
  titleFont: { family: FONT_FAMILY, size: 13, weight: '600' as const },
  bodyFont: { family: FONT_FAMILY, size: 12 },
  cornerRadius: 10,
  padding: { top: 10, bottom: 10, left: 14, right: 14 },
  displayColors: true,
  boxPadding: 6,
  caretSize: 6,
  caretPadding: 8,
  borderColor: 'rgba(255,255,255,0.08)',
  borderWidth: 1,
  usePointStyle: true,
} as const;

// --- Gradient helpers ---

/** Create a vertical gradient from top (colorTop) to bottom (colorBottom) */
export function verticalGradient(
  ctx: CanvasRenderingContext2D,
  chartArea: { top: number; bottom: number },
  colorTop: string,
  colorBottom: string
): CanvasGradient {
  const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  g.addColorStop(0, colorTop);
  g.addColorStop(1, colorBottom);
  return g;
}

/** Create a horizontal gradient */
export function horizontalGradient(
  ctx: CanvasRenderingContext2D,
  chartArea: { left: number; right: number },
  colorLeft: string,
  colorRight: string
): CanvasGradient {
  const g = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
  g.addColorStop(0, colorLeft);
  g.addColorStop(1, colorRight);
  return g;
}

// --- Animation helpers ---

/** Staggered animation delay based on data index */
export function staggerDelay(ctx: any): number {
  return ctx.dataIndex * 60 + ctx.datasetIndex * 100;
}

/** Default animation config with staggered delay */
export const ANIMATION_DEFAULTS = {
  duration: 900,
  easing: 'easeOutQuart' as const,
  delay: staggerDelay,
};

// --- Interaction defaults ---
export const INTERACTION_DEFAULTS = {
  mode: 'nearest' as const,
  intersect: false,
  axis: 'xy' as const,
};

// --- Crosshair plugin ---
export const crosshairPlugin = {
  id: 'crosshair',
  afterDraw(chart: any) {
    const tooltip = chart.tooltip;
    if (!tooltip || !tooltip.opacity) return;

    const ctx = chart.ctx;
    const chartArea = chart.chartArea;
    const x = tooltip.caretX;

    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.dim;
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

// --- Hover highlight plugin (dims non-hovered datasets) ---
export const hoverHighlightPlugin = {
  id: 'hoverHighlight',
  beforeDraw(chart: any) {
    const tooltip = chart.tooltip;
    if (!tooltip || !tooltip.opacity) {
      // Reset all datasets to full opacity
      chart.data.datasets.forEach((_ds: any, i: number) => {
        chart.getDatasetMeta(i).hidden = false;
      });
      return;
    }

    const activeDatasetIndex = tooltip.dataPoints?.[0]?.datasetIndex;
    if (activeDatasetIndex == null) return;

    chart.data.datasets.forEach((ds: any, i: number) => {
      const meta = chart.getDatasetMeta(i);
      meta.dataset && (meta.dataset.options = meta.dataset.options || {});
      if (i !== activeDatasetIndex) {
        meta.dataset && (meta.dataset.options.borderWidth = 1.5);
      }
    });
  },
};

// --- Scale defaults for cleaner grids ---
export const SCALE_X_DEFAULTS = {
  grid: { display: false },
  border: { display: false },
  ticks: {
    font: { family: FONT_FAMILY, size: 11 },
    color: COLORS.inkSoft,
    padding: 8,
  },
};

export const SCALE_Y_DEFAULTS = {
  grid: {
    color: 'rgba(240,242,245,0.7)',
    lineWidth: 1,
    drawTicks: false,
  },
  border: { display: false, dash: [4, 4] },
  ticks: {
    font: { family: FONT_FAMILY, size: 11 },
    color: COLORS.muted,
    padding: 10,
  },
};
