// Tree-shaken Chart.js registration
import {
  Chart,
  BarController,
  LineController,
  BarElement,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import DataLabelsPlugin from 'chartjs-plugin-datalabels';
import { crosshairPlugin } from './theme';

Chart.register(
  BarController,
  LineController,
  BarElement,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin,
  DataLabelsPlugin,
  crosshairPlugin as any,
);

// Global defaults for cleaner look
Chart.defaults.font.family = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";
Chart.defaults.plugins.datalabels = { display: false }; // off by default, enabled per-chart

export { Chart };
