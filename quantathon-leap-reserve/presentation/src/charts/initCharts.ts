import { Chart } from './register';
import { buildHistogramConfig } from './histogram';
import { buildGroupedBarConfig } from './groupedBar';
import { buildComboBarLineConfig } from './comboBarLine';
import { buildDualLineConfig } from './dualLine';
import { buildTornadoConfig } from './tornado';
import { buildRecoveryLineConfig } from './recoveryLine';
import { buildStressBarConfig } from './stressBar';
import { buildHorizontalBarConfig } from './horizontalBar';
import { buildFunnelConfig } from './funnel';
import type { ChartConfiguration } from 'chart.js';

type ConfigBuilder = (data: any) => ChartConfiguration<any>;

const builders: Record<string, ConfigBuilder> = {
  histogram: buildHistogramConfig,
  groupedBar: buildGroupedBarConfig,
  comboBarLine: buildComboBarLineConfig,
  dualLine: buildDualLineConfig,
  tornado: buildTornadoConfig,
  recoveryLine: buildRecoveryLineConfig,
  stressBar: buildStressBarConfig,
  horizontalBar: buildHorizontalBarConfig,
  funnel: buildFunnelConfig,
};

const chartInstances = new Map<string, Chart>();

function initChart(canvas: HTMLCanvasElement) {
  const chartType = canvas.dataset.chartType;
  if (!chartType || !builders[chartType]) return;

  const jsonEl = canvas.parentElement?.querySelector('script[type="application/json"]');
  if (!jsonEl) return;

  let data: any;
  try {
    data = JSON.parse(jsonEl.textContent || '{}');
  } catch {
    return;
  }

  const config = builders[chartType](data);
  const id = canvas.id || chartType;

  // Destroy existing instance if any
  if (chartInstances.has(id)) {
    chartInstances.get(id)!.destroy();
  }

  const chart = new Chart(canvas, config);
  chartInstances.set(id, chart);
}

function initAll() {
  document.querySelectorAll<HTMLCanvasElement>('canvas[data-chart-type]').forEach(initChart);
}

// Initialize charts immediately
initAll();

// Re-trigger animations when a slide becomes active
const observer = new MutationObserver((mutations) => {
  for (const m of mutations) {
    if (m.type === 'attributes' && m.attributeName === 'class') {
      const slide = m.target as HTMLElement;
      if (slide.classList.contains('active') && slide.classList.contains('slide')) {
        const canvases = slide.querySelectorAll<HTMLCanvasElement>('canvas[data-chart-type]');
        canvases.forEach(c => {
          const id = c.id || c.dataset.chartType || '';
          const chart = chartInstances.get(id);
          if (chart) {
            chart.reset();
            chart.update('active');
          }
        });
      }
    }
  }
});

document.querySelectorAll('.slide').forEach(slide => {
  observer.observe(slide, { attributes: true, attributeFilter: ['class'] });
});
