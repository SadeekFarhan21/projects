import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface FunnelData {
  labels: string[];
  values: number[];
}

export function buildFunnelConfig(data: FunnelData): ChartConfiguration<'bar'> {
  const maxVal = Math.max(...data.values);

  return {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          data: data.values,
          backgroundColor: data.values.map((v) => {
            const ratio = v / maxVal;
            if (ratio > 0.8) return 'rgba(43,82,180,0.18)';
            if (ratio > 0.5) return 'rgba(43,82,180,0.28)';
            return 'rgba(43,82,180,0.38)';
          }),
          hoverBackgroundColor: data.values.map((v) => {
            const ratio = v / maxVal;
            if (ratio > 0.8) return 'rgba(43,82,180,0.28)';
            if (ratio > 0.5) return 'rgba(43,82,180,0.40)';
            return 'rgba(43,82,180,0.52)';
          }),
          borderColor: data.values.map((v) => {
            const ratio = v / maxVal;
            if (ratio > 0.8) return 'rgba(43,82,180,0.25)';
            if (ratio > 0.5) return 'rgba(43,82,180,0.40)';
            return COLORS.primary;
          }),
          borderWidth: { left: 3, top: 0, right: 0, bottom: 0 },
          borderRadius: { topRight: 6, bottomRight: 6, topLeft: 0, bottomLeft: 0 },
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1000,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 150; },
      },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            title: (items) => items[0]?.label || '',
            label: (item) => {
              const val = item.parsed.x;
              const pct = ((val / maxVal) * 100).toFixed(0);
              return ` ${val.toLocaleString()} (${pct}% of top)`;
            },
          },
        },
        datalabels: {
          display: true,
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.ink,
          font: { family: FONT_FAMILY, size: 13, weight: 'bold' as const },
          formatter: (v: number) => v.toLocaleString(),
          offset: 6,
        },
      },
      scales: {
        x: {
          ...SCALE_X_DEFAULTS,
          display: false,
          grid: { display: false },
        },
        y: {
          ...SCALE_Y_DEFAULTS,
          grid: { display: false },
          ticks: {
            ...SCALE_Y_DEFAULTS.ticks,
            color: COLORS.inkSoft,
            font: { family: FONT_FAMILY, size: 12, weight: '500' },
          },
        },
      },
    },
  };
}
