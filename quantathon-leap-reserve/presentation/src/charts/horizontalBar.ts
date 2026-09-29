import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface HorizontalBarData {
  labels: string[];
  values: number[];
  unit?: 'currency' | 'percent';
}

export function buildHorizontalBarConfig(data: HorizontalBarData): ChartConfiguration<'bar'> {
  const isCurrency = data.unit !== 'percent';
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
            return ratio > 0.8
              ? 'rgba(43,82,180,0.75)'
              : ratio > 0.5
                ? 'rgba(233,122,46,0.70)'
                : 'rgba(139,184,240,0.65)';
          }),
          hoverBackgroundColor: data.values.map((v) => {
            const ratio = v / maxVal;
            return ratio > 0.8 ? COLORS.primary : ratio > 0.5 ? COLORS.accent : COLORS.primaryLight;
          }),
          borderColor: data.values.map((v) => {
            const ratio = v / maxVal;
            return ratio > 0.8 ? COLORS.primaryDeep : ratio > 0.5 ? COLORS.accentDeep : COLORS.primary;
          }),
          borderWidth: 1,
          borderRadius: 6,
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
        delay(ctx: any) { return ctx.dataIndex * 80; },
      },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            label: (item) => isCurrency
              ? ` ${currencyFormatter(item.parsed.x)}`
              : ` ${(item.parsed.x * 100).toFixed(1)}%`,
          },
        },
        datalabels: {
          display: true,
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.inkSoft,
          font: { family: FONT_FAMILY, size: 10, weight: 'bold' as const },
          formatter: (v: number) => isCurrency ? currencyFormatter(v) : `${(v * 100).toFixed(1)}%`,
          offset: 4,
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
            font: { family: FONT_FAMILY, size: 10 },
            color: COLORS.muted,
            callback: (v: any) => isCurrency
              ? currencyFormatter(v as number)
              : `${((v as number) * 100).toFixed(0)}%`,
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
