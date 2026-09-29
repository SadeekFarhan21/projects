import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, pctFormatter, TOOLTIP_DEFAULTS, verticalGradient, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface RecoveryEntry {
  calendar_year: number;
  cumulative_recovery_fraction: number;
}

export interface RecoveryData {
  projection: RecoveryEntry[];
  milestones: Record<string, { year: number; years_after_2037: number }>;
}

export function buildRecoveryLineConfig(data: RecoveryData): ChartConfiguration<'line'> {
  const show = data.projection.slice(0, 25);
  const labels = show.map(e => String(e.calendar_year));
  const values = show.map(e => e.cumulative_recovery_fraction);

  const annotations: Record<string, any> = {};
  if (data.milestones['80%']) {
    annotations.line80 = {
      type: 'line',
      yMin: 0.8,
      yMax: 0.8,
      borderColor: COLORS.accentDeep,
      borderWidth: 1.5,
      borderDash: [4, 4],
      label: {
        display: true,
        content: `80% → ${data.milestones['80%'].year}`,
        position: { x: 'end', y: 'end' },
        yAdjust: -14,
        backgroundColor: COLORS.accentDeep,
        color: '#fff',
        font: { family: FONT_FAMILY, size: 10, weight: '600' },
        padding: { top: 3, bottom: 3, left: 8, right: 8 },
        borderRadius: 5,
      },
    };
  }
  if (data.milestones['90%']) {
    annotations.line90 = {
      type: 'line',
      yMin: 0.9,
      yMax: 0.9,
      borderColor: COLORS.purple,
      borderWidth: 1.5,
      borderDash: [4, 4],
      label: {
        display: true,
        content: `90% → ${data.milestones['90%'].year}`,
        position: { x: 'end', y: 'end' },
        yAdjust: -14,
        backgroundColor: COLORS.purple,
        color: '#fff',
        font: { family: FONT_FAMILY, size: 10, weight: '600' },
        padding: { top: 3, bottom: 3, left: 8, right: 8 },
        borderRadius: 5,
      },
    };
  }

  return {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Cumulative Recovery',
        data: values,
        borderColor: COLORS.primary,
        borderWidth: 3,
        backgroundColor(ctx: any) {
          const chart = ctx.chart;
          const { ctx: canvasCtx, chartArea } = chart;
          if (!chartArea) return COLORS.primaryGlow;
          return verticalGradient(canvasCtx, chartArea, 'rgba(43,82,180,0.20)', 'rgba(43,82,180,0.02)');
        },
        fill: true,
        pointBackgroundColor: COLORS.surface,
        pointBorderColor: COLORS.primary,
        pointBorderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 7,
        pointHoverBackgroundColor: COLORS.primary,
        pointHoverBorderColor: COLORS.surface,
        pointHoverBorderWidth: 3,
        tension: 0.35,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1200,
        easing: 'easeOutQuart',
      },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            title: (items) => `Year ${items[0]?.label}`,
            label: (item) => ` Recovery: ${pctFormatter(item.parsed.y)}`,
          },
        },
        datalabels: { display: false },
        annotation: { annotations },
      },
      scales: {
        x: {
          ...SCALE_X_DEFAULTS,
          ticks: {
            ...SCALE_X_DEFAULTS.ticks,
            font: { family: FONT_FAMILY, size: 10 },
            color: COLORS.muted,
            maxTicksLimit: 10,
          },
        },
        y: {
          min: 0,
          max: 1,
          ...SCALE_Y_DEFAULTS,
          ticks: {
            ...SCALE_Y_DEFAULTS.ticks,
            callback: (v: any) => pctFormatter(v as number),
          },
          title: {
            display: true,
            text: 'Cumulative Recovery',
            font: { family: FONT_FAMILY, size: 11, weight: '600' },
            color: COLORS.inkSoft,
          },
        },
      },
    },
  };
}
