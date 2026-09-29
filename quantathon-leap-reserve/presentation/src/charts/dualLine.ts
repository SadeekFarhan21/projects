import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface DualLineData {
  labels: string[];
  mcOutflows: number[];
  mcInflows: number[];
  detOutflows: number[];
  detInflows: number[];
}

export function buildDualLineConfig(data: DualLineData): ChartConfiguration<'line'> {
  return {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'MC Outflows',
          data: data.mcOutflows,
          borderColor: COLORS.primary,
          borderWidth: 3,
          backgroundColor: 'rgba(43,82,180,0.06)',
          fill: true,
          pointBackgroundColor: COLORS.surface,
          pointBorderColor: COLORS.primary,
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointHoverBackgroundColor: COLORS.primary,
          pointHoverBorderColor: COLORS.surface,
          pointHoverBorderWidth: 2,
          tension: 0.35,
        },
        {
          label: 'Det. Outflows',
          data: data.detOutflows,
          borderColor: COLORS.primary,
          borderWidth: 2,
          borderDash: [8, 4],
          pointBackgroundColor: COLORS.surface,
          pointBorderColor: COLORS.primary,
          pointBorderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: COLORS.primary,
          pointStyle: 'triangle',
          tension: 0.35,
          fill: false,
        },
        {
          label: 'MC Inflows',
          data: data.mcInflows,
          borderColor: COLORS.red,
          borderWidth: 3,
          backgroundColor: 'rgba(220,38,38,0.06)',
          fill: true,
          pointBackgroundColor: COLORS.surface,
          pointBorderColor: COLORS.red,
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointHoverBackgroundColor: COLORS.red,
          pointHoverBorderColor: COLORS.surface,
          pointHoverBorderWidth: 2,
          tension: 0.35,
        },
        {
          label: 'Det. Inflows',
          data: data.detInflows,
          borderColor: COLORS.red,
          borderWidth: 2,
          borderDash: [8, 4],
          pointBackgroundColor: COLORS.surface,
          pointBorderColor: COLORS.red,
          pointBorderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: COLORS.red,
          pointStyle: 'triangle',
          tension: 0.35,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1000,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.datasetIndex * 200; },
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
          intersect: false,
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
          ...SCALE_Y_DEFAULTS,
          ticks: {
            ...SCALE_Y_DEFAULTS.ticks,
            callback: (v: any) => currencyFormatter(v as number),
          },
        },
      },
    },
  };
}
