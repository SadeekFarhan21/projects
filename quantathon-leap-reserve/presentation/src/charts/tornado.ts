import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface TornadoParam {
  parameter: string;
  optimistic_p95: number;
  conservative_p95: number;
  swing: number;
}

export interface TornadoData {
  parameters: TornadoParam[];
  base_p95: number;
}

export function buildTornadoConfig(data: TornadoData): ChartConfiguration<'bar'> {
  const params = data.parameters;
  const labels = params.map(p => p.parameter);

  const barData = params.map(p => {
    const lo = Math.min(p.optimistic_p95, p.conservative_p95);
    const hi = Math.max(p.optimistic_p95, p.conservative_p95);
    return [lo, hi];
  });

  return {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'P95 Range',
        data: barData,
        backgroundColor: params.map((_, i) => {
          const ratio = 1 - i / params.length;
          const alpha = 0.35 + 0.45 * ratio;
          return `rgba(43,82,180,${alpha})`;
        }),
        hoverBackgroundColor: params.map((_, i) => {
          const ratio = 1 - i / params.length;
          const alpha = 0.5 + 0.4 * ratio;
          return `rgba(43,82,180,${alpha})`;
        }),
        borderColor: COLORS.primary,
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 100; },
      },
      interaction: { mode: 'index', intersect: false, axis: 'y' },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            title: (items) => items[0]?.label || '',
            label: (item) => {
              const param = params[item.dataIndex];
              return [
                ` Optimistic: ${currencyFormatter(param.optimistic_p95)}`,
                ` Conservative: ${currencyFormatter(param.conservative_p95)}`,
                ` Swing: ${currencyFormatter(Math.abs(param.swing))}`,
              ];
            },
          },
        },
        datalabels: {
          display: true,
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.muted,
          font: { family: FONT_FAMILY, size: 9, weight: 'bold' as const },
          formatter: (_v: any, ctx: any) => {
            const param = params[ctx.dataIndex];
            return currencyFormatter(Math.abs(param.swing));
          },
          offset: 4,
        },
        annotation: {
          annotations: {
            baseLine: {
              type: 'line',
              xMin: data.base_p95,
              xMax: data.base_p95,
              borderColor: COLORS.inkSoft,
              borderWidth: 2,
              borderDash: [6, 4],
              label: {
                display: true,
                content: `Base: ${currencyFormatter(data.base_p95)}`,
                position: 'start',
                backgroundColor: COLORS.inkSoft,
                color: '#fff',
                font: { family: FONT_FAMILY, size: 11, weight: '600' },
                padding: { top: 4, bottom: 4, left: 8, right: 8 },
                borderRadius: 6,
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
            font: { family: FONT_FAMILY, size: 10 },
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
