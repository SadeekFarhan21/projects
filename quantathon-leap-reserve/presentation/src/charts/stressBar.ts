import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface StressBarData {
  labels: string[];
  values: number[];
  baseline: number;
}

export function buildStressBarConfig(data: StressBarData): ChartConfiguration<'bar'> {
  return {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'P95 Funding Required',
          data: data.values,
          backgroundColor: data.values.map((_v, i) =>
            i === 0 ? 'rgba(43,82,180,0.75)' : i === data.values.length - 1 ? 'rgba(220,38,38,0.75)' : 'rgba(233,122,46,0.75)'
          ),
          hoverBackgroundColor: data.values.map((_v, i) =>
            i === 0 ? COLORS.primary : i === data.values.length - 1 ? COLORS.red : COLORS.accent
          ),
          borderColor: data.values.map((_v, i) =>
            i === 0 ? COLORS.primaryDeep : i === data.values.length - 1 ? COLORS.red : COLORS.accentDeep
          ),
          borderWidth: 1,
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 120; },
      },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            label: (item) => ` P95: ${currencyFormatter(item.parsed.x)}`,
          },
        },
        datalabels: {
          display: true,
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.inkSoft,
          font: { family: FONT_FAMILY, size: 10, weight: 'bold' as const },
          formatter: (v: number) => currencyFormatter(v),
          offset: 4,
        },
        annotation: {
          annotations: {
            baseline: {
              type: 'line',
              xMin: data.baseline,
              xMax: data.baseline,
              borderColor: COLORS.primary,
              borderWidth: 2,
              borderDash: [6, 4],
              label: {
                display: true,
                content: `Base ${currencyFormatter(data.baseline)}`,
                position: { x: 'center', y: 'start' },
                yAdjust: -20,
                font: { family: FONT_FAMILY, size: 10, weight: '600' },
                color: COLORS.primary,
                backgroundColor: 'rgba(255,255,255,0.92)',
                padding: { top: 4, bottom: 4, left: 8, right: 8 },
                borderRadius: 5,
              },
            },
          },
        },
      },
      scales: {
        x: {
          ...SCALE_X_DEFAULTS,
          grid: {
            color: 'rgba(240,242,245,0.7)',
            lineWidth: 1,
            drawTicks: false,
          },
          ticks: {
            ...SCALE_X_DEFAULTS.ticks,
            color: COLORS.muted,
            callback: (v: any) => currencyFormatter(v as number),
          },
        },
        y: {
          ...SCALE_Y_DEFAULTS,
          grid: { display: false },
          ticks: {
            ...SCALE_Y_DEFAULTS.ticks,
            color: COLORS.inkSoft,
            font: { family: FONT_FAMILY, size: 11, weight: '500' },
          },
        },
      },
    },
  };
}
