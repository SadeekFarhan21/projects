import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, verticalGradient, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface ComboBarLineData {
  labels: string[];
  outflows: number[];
  inflows: number[];
  cumulativeInflows: number[];
}

export function buildComboBarLineConfig(data: ComboBarLineData): ChartConfiguration {
  return {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          type: 'bar' as const,
          label: 'Annual Outflows',
          data: data.outflows,
          backgroundColor: 'rgba(43,82,180,0.65)',
          borderColor: COLORS.primary,
          borderWidth: 1.5,
          borderRadius: 6,
          hoverBackgroundColor: 'rgba(43,82,180,0.85)',
          hoverBorderWidth: 2,
          order: 2,
          yAxisID: 'y',
        },
        {
          type: 'bar' as const,
          label: 'Annual Repayments',
          data: data.inflows,
          backgroundColor: 'rgba(233,122,46,0.65)',
          borderColor: COLORS.accent,
          borderWidth: 1.5,
          borderRadius: 6,
          hoverBackgroundColor: 'rgba(233,122,46,0.85)',
          hoverBorderWidth: 2,
          order: 2,
          yAxisID: 'y',
        },
        {
          type: 'line' as const,
          label: 'Cumulative Repayments',
          data: data.cumulativeInflows,
          borderColor: COLORS.accentDeep,
          borderWidth: 3,
          pointBackgroundColor: COLORS.surface,
          pointBorderColor: COLORS.accentDeep,
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointHoverBackgroundColor: COLORS.accentDeep,
          pointHoverBorderColor: COLORS.surface,
          pointHoverBorderWidth: 2,
          tension: 0.35,
          fill: {
            target: 'origin',
            above: 'rgba(196,90,18,0.06)',
          },
          order: 1,
          yAxisID: 'y2',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 60 + ctx.datasetIndex * 120; },
      },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { family: FONT_FAMILY, size: 11, weight: '500' },
            color: COLORS.inkSoft,
            usePointStyle: true,
            padding: 18,
          },
        },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          mode: 'index',
          callbacks: {
            label: (item) => ` ${item.dataset.label}: ${currencyFormatter(item.parsed.y)}`,
          },
        },
        datalabels: { display: false },
      },
      scales: {
        x: {
          ...SCALE_X_DEFAULTS,
        },
        y: {
          position: 'left',
          ...SCALE_Y_DEFAULTS,
          ticks: {
            ...SCALE_Y_DEFAULTS.ticks,
            callback: (v: any) => currencyFormatter(v as number),
          },
          title: {
            display: true,
            text: 'Annual Amount',
            font: { family: FONT_FAMILY, size: 11, weight: '600' },
            color: COLORS.inkSoft,
          },
        },
        y2: {
          position: 'right',
          grid: { drawOnChartArea: false },
          border: { display: false },
          ticks: {
            font: { family: FONT_FAMILY, size: 10 },
            color: COLORS.accentDeep,
            callback: (v: any) => currencyFormatter(v as number),
            padding: 10,
          },
          title: {
            display: true,
            text: 'Cumulative Repayments',
            font: { family: FONT_FAMILY, size: 11, weight: '600' },
            color: COLORS.accentDeep,
          },
        },
      },
    },
  };
}
