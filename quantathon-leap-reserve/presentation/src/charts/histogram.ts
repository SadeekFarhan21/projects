import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, verticalGradient, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface HistogramData {
  bin_edges: number[];
  counts: number[];
  mean: number;
}

export function buildHistogramConfig(data: HistogramData): ChartConfiguration<'bar'> {
  const labels = data.bin_edges.slice(0, -1).map((edge, i) => {
    const next = data.bin_edges[i + 1];
    return `${currencyFormatter(edge)} – ${currencyFormatter(next)}`;
  });

  const maxCount = Math.max(...data.counts);

  return {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Simulations',
        data: data.counts,
        backgroundColor(ctx: any) {
          const chart = ctx.chart;
          const { ctx: canvasCtx, chartArea } = chart;
          if (!chartArea) return COLORS.primaryGlow;
          return verticalGradient(canvasCtx, chartArea, 'rgba(43,82,180,0.35)', 'rgba(43,82,180,0.08)');
        },
        borderColor: COLORS.primary,
        borderWidth: 1.5,
        borderRadius: 6,
        hoverBackgroundColor: 'rgba(43,82,180,0.45)',
        hoverBorderColor: COLORS.primaryDeep,
        hoverBorderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 50; },
      },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            title: (items) => items[0]?.label || '',
            label: (item) => ` ${item.parsed.y.toLocaleString()} simulations`,
          },
        },
        datalabels: {
          display(ctx: any) {
            return ctx.dataset.data[ctx.dataIndex] > maxCount * 0.6;
          },
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.inkSoft,
          font: { family: FONT_FAMILY, size: 10, weight: 'bold' as const },
          formatter: (v: number) => v.toLocaleString(),
          offset: 2,
        },
        annotation: {
          annotations: {
            meanLine: {
              type: 'line',
              xMin: (() => {
                for (let i = 0; i < data.bin_edges.length - 1; i++) {
                  if (data.mean >= data.bin_edges[i] && data.mean < data.bin_edges[i + 1]) {
                    const frac = (data.mean - data.bin_edges[i]) / (data.bin_edges[i + 1] - data.bin_edges[i]);
                    return i + frac;
                  }
                }
                return data.bin_edges.length - 2;
              })(),
              xMax: (() => {
                for (let i = 0; i < data.bin_edges.length - 1; i++) {
                  if (data.mean >= data.bin_edges[i] && data.mean < data.bin_edges[i + 1]) {
                    const frac = (data.mean - data.bin_edges[i]) / (data.bin_edges[i + 1] - data.bin_edges[i]);
                    return i + frac;
                  }
                }
                return data.bin_edges.length - 2;
              })(),
              borderColor: COLORS.accent,
              borderWidth: 2,
              borderDash: [6, 4],
              label: {
                display: true,
                content: `Mean: ${currencyFormatter(data.mean)}`,
                position: 'start',
                backgroundColor: COLORS.accentDeep,
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
          ticks: {
            ...SCALE_X_DEFAULTS.ticks,
            font: { family: FONT_FAMILY, size: 9 },
            color: COLORS.muted,
            maxRotation: 45,
            minRotation: 45,
          },
        },
        y: {
          ...SCALE_Y_DEFAULTS,
          title: {
            display: true,
            text: 'Simulations',
            font: { family: FONT_FAMILY, size: 11, weight: '600' },
            color: COLORS.inkSoft,
          },
        },
      },
    },
  };
}
