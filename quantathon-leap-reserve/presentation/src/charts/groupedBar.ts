import type { ChartConfiguration } from 'chart.js';
import { COLORS, FONT_FAMILY, currencyFormatter, TOOLTIP_DEFAULTS, SCALE_X_DEFAULTS, SCALE_Y_DEFAULTS } from './theme';

export interface GroupedBarData {
  labels: string[];
  loan: number[];
  grant: number[];
}

export function buildGroupedBarConfig(data: GroupedBarData): ChartConfiguration<'bar'> {
  return {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'Loan Model',
          data: data.loan,
          backgroundColor: 'rgba(43,82,180,0.75)',
          borderColor: COLORS.primaryDeep,
          borderWidth: 1,
          borderRadius: 8,
          hoverBackgroundColor: COLORS.primary,
          hoverBorderWidth: 2,
        },
        {
          label: 'Grant Model',
          data: data.grant,
          backgroundColor: 'rgba(233,122,46,0.75)',
          borderColor: COLORS.accentDeep,
          borderWidth: 1,
          borderRadius: 8,
          hoverBackgroundColor: COLORS.accent,
          hoverBorderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: 'easeOutQuart',
        delay(ctx: any) { return ctx.dataIndex * 80 + ctx.datasetIndex * 150; },
      },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { family: FONT_FAMILY, size: 11, weight: '500' },
            color: COLORS.inkSoft,
            usePointStyle: true,
            pointStyle: 'rectRounded',
            padding: 20,
          },
        },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          mode: 'index',
          callbacks: {
            label: (item) => ` ${item.dataset.label}: ${currencyFormatter(item.parsed.y)}`,
          },
        },
        datalabels: {
          display(ctx: any) { return ctx.dataset.data[ctx.dataIndex] > 0; },
          anchor: 'end' as const,
          align: 'end' as const,
          color: COLORS.inkSoft,
          font: { family: FONT_FAMILY, size: 9, weight: 'bold' as const },
          formatter: (v: number) => currencyFormatter(v),
          offset: 2,
        },
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
          title: {
            display: true,
            text: 'Required Funding',
            font: { family: FONT_FAMILY, size: 11, weight: '600' },
            color: COLORS.inkSoft,
          },
        },
      },
    },
  };
}
