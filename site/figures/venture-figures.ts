/**
 * D3 figures for the venture memos.
 *
 * A memo drops a placeholder into its markdown:
 *
 *   <figure data-figure="venture:wafer-capital"></figure>
 *
 * and this module renders /data/ventures/wafer-capital.json into it. Every
 * chart is a JSON spec rather than code, so the numbers live next to their
 * source line and a figure can be audited without reading TypeScript.
 *
 * Forms:
 *   bars      horizontal bars, one series, optional log axis and emphasis row
 *   columns   grouped or stacked columns over categories, with forecast styling
 *   dots      a dot plot on one shared axis, measured vs claimed classes
 *   series    sparse dated points joined per entity, never smoothed
 *   timeline  dated events in lanes, shipped vs announced markers
 *   matrix    a disclosure grid: what each row has published
 *   curve     a sensitivity model with a slider, explicitly not data
 */
import { select } from "d3-selection";
import { scaleLinear, scaleLog, scaleBand } from "d3-scale";
import { tickStep } from "d3-array";
import {
  theme,
  SERIES,
  DEEMPHASIS,
  seriesColor,
  fmt,
  showTip,
  hideTip,
  caption,
  sourceLine,
  heading,
  legend,
  responsive,
  textWidth,
  fitText,
  wrapText,
  setLines,
  spacedTicks,
  clampAnchor,
  json,
  type Layout,
} from "./figure-kit";

/* ------------------------------------------------------------- formatting */

type Format = "money" | "percent" | "number" | "multiple" | "time";

/** Compact money: $44M, $2.1B, $13B. */
function money(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `$${trim(v / 1e9)}B`;
  if (abs >= 1e6) return `$${trim(v / 1e6)}M`;
  if (abs >= 1e3) return `$${trim(v / 1e3)}K`;
  return v === 0 ? "$0" : `$${v.toFixed(2)}`;
}
function trim(n: number): string {
  const s = n.toFixed(n >= 100 ? 0 : n >= 10 ? 1 : 2);
  return s.replace(/\.0+$/, "").replace(/(\.\d*[1-9])0+$/, "$1");
}
/** Durations given in seconds: 50 ms, 300 µs, 1 µs. */
function duration(s: number): string {
  // A bare 0 on a µs or ms axis: "0 ns" read as a unit switch.
  if (s === 0) return "0";
  if (s >= 1) return `${trim(s)} s`;
  if (s >= 1e-3) return `${trim(s * 1e3)} ms`;
  if (s >= 1e-6) return `${trim(s * 1e6)} µs`;
  return `${trim(s * 1e9)} ns`;
}
function format(v: number, f: Format = "number"): string {
  switch (f) {
    case "money":
      return money(v);
    case "percent":
      return `${trim(v)}%`;
    case "multiple":
      return `${trim(v)}×`;
    case "time":
      return duration(v);
    default:
      return fmt.format(v);
  }
}

/**
 * A linear value axis whose ends are ticks: about `count` round steps from
 * the tick at or below lo to the one at or above hi, so the top gridline is
 * never under the tallest mark (d3's nice() rounds to a finer step than
 * ticks(4) then labels, which left bars past the last labelled line).
 */
function linearAxis(lo: number, hi: number, count: number) {
  const step = tickStep(lo, hi, count) || 1;
  const a = Math.floor(lo / step) * step,
    b = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toPrecision(12));
  return { domain: [a, b] as [number, number], ticks };
}

/** Log ticks at 1, 10, 100 … within the domain. */
function logTicks(lo: number, hi: number): number[] {
  for (const mants of [[1, 2, 5], [1, 5], [1]]) {
    const out: number[] = [];
    for (
      let e = Math.floor(Math.log10(lo));
      e <= Math.ceil(Math.log10(hi));
      e++
    )
      for (const m of mants) {
        const v = m * 10 ** e;
        if (v >= lo && v <= hi) out.push(v);
      }
    if (out.length <= 6) return out;
  }
  return [];
}

/* ---------------------------------------------------------------- chrome */

type Base = {
  title: string;
  subtitle?: string;
  caption: string;
  source: string;
};

function footer(node: Element, spec: Base) {
  caption(node, spec.caption);
  sourceLine(node, spec.source);
}

type Svg = ReturnType<typeof select<SVGSVGElement, unknown>>;

/**
 * Size a chart's responsive <svg> (figure-kit responsive()) to w x h drawing
 * units, which are CSS px: the viewBox matches the figure's content width.
 */
function sized(el: SVGSVGElement, w: number, h: number, label: string): Svg {
  return select(el)
    .attr("viewBox", `0 0 ${w} ${h}`)
    .attr("role", "img")
    .attr("aria-label", label) as unknown as Svg;
}

/** Widest of some labels at a font size, in px. */
const widest = (labels: string[], px: number, weight: number | string = 400) =>
  Math.max(0, ...labels.map(l => textWidth(l, px, weight)));

/**
 * An axis title's lines: word-wrapped to two lines of w rather than cut, so a
 * long title ("..., log scale") keeps its last words on a phone. Callers
 * reserve (lines - 1) * titleLineH of extra height.
 */
function titleLines(text: string, w: number, px: number): string[] {
  return wrapText(text, w, px, 2);
}
const titleLineH = (px: number) => px + 3;

/**
 * An axis title (the spec's unit or xLabel) in the muted axis style, wrapped
 * to the room it has. Centred on cx but kept inside [0, w]; with anchor
 * "start" it sits flush left at cx.
 */
function axisTitle(
  svg: Svg,
  text: string,
  cx: number,
  y: number,
  w: number,
  px: number,
  anchor?: "start"
) {
  const lines = titleLines(text, w, px);
  const c = anchor
    ? { x: cx, anchor }
    : clampAnchor(cx, widest(lines, px), 0, w);
  const el = svg
    .append("text")
    .attr("class", "fig-axis fig-axis-title")
    .attr("x", c.x)
    .attr("y", y)
    .attr("text-anchor", c.anchor);
  setLines(el.node() as SVGTextElement, lines, c.x, titleLineH(px));
}

/** Does the segment p-q cross the box [x1,x2]x[y1,y2]? (Liang-Barsky clip.) */
function segHitsBox(
  p: [number, number],
  q: [number, number],
  x1: number,
  y1: number,
  x2: number,
  y2: number
): boolean {
  let t0 = 0,
    t1 = 1;
  const dx = q[0] - p[0],
    dy = q[1] - p[1];
  const edges: [number, number][] = [
    [-dx, p[0] - x1],
    [dx, x2 - p[0]],
    [-dy, p[1] - y1],
    [dy, y2 - p[1]],
  ];
  for (const [pp, qq] of edges) {
    if (pp === 0) {
      if (qq < 0) return false;
    } else {
      const r = qq / pp;
      if (pp < 0) t0 = Math.max(t0, r);
      else t1 = Math.min(t1, r);
      if (t0 > t1) return false;
    }
  }
  return true;
}

function svgIn(node: Element, w: number, h: number, label: string): Svg {
  return select(node)
    .append("svg")
    .attr("viewBox", `0 0 ${w} ${h}`)
    .attr("width", "100%")
    .attr("role", "img")
    .attr("aria-label", label) as unknown as Svg;
}

/** Diagonal hatch used for every forecast, guidance or target mark. */
function hatch(svg: Svg, id: string, color: string) {
  const p = svg
    .append("defs")
    .append("pattern")
    .attr("id", id)
    .attr("patternUnits", "userSpaceOnUse")
    .attr("width", 6)
    .attr("height", 6)
    .attr("patternTransform", "rotate(45)");
  p.append("line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 0)
    .attr("y2", 6)
    .attr("stroke", color)
    .attr("stroke-width", 1.5);
  return `url(#${id})`;
}

/* ------------------------------------------------------------------ bars */

type BarRow = {
  label: string;
  /** null means the value is not disclosed; the row is kept, the bar is not drawn. */
  value: number | null;
  /** Short text after the value: "Series F, Jun 2026". */
  note?: string;
  /** Tooltip detail. */
  detail?: string;
  /** The subject of the memo: drawn in the accent, everything else in gray. */
  emphasis?: boolean;
  /** A forecast, target or author estimate: hatched rather than filled. */
  estimate?: boolean;
};

type BarsSpec = Base & {
  type: "bars";
  rows: BarRow[];
  format?: Format;
  scale?: "linear" | "log";
  unit?: string;
  /** A reference line, e.g. category revenue. */
  reference?: { value: number; label: string; band?: boolean };
  /** Legend wording for accent and gray rows; defaults suit the memos. */
  legendLabels?: { emphasis?: string; other?: string };
};

function bars(node: Element, d: BarsSpec) {
  const t = theme();
  const f = d.format ?? "number";
  const values = d.rows
    .map(r => r.value)
    .filter((v): v is number => v !== null);
  const maxV = Math.max(...values, d.reference?.value ?? 0);
  const minV = Math.min(...values);
  // Negative values (a linear axis only) grow bars leftward from a zero line,
  // with room past the most negative bar for its value label. All-positive
  // specs keep the original [0, max] domain.
  const neg = d.scale !== "log" && minV < 0;
  const span = Math.max(maxV, 0) - minV;
  const domain: [number, number] =
    d.scale === "log"
      ? [10 ** Math.floor(Math.log10(minV)), maxV * 1.15]
      : neg
        ? [minV - span * 0.2, Math.max(maxV, 0) + span * 0.08]
        : [0, maxV * 1.08];
  // Value only at the bar end; the tooltip has the note and the detail.
  const endText = (r: BarRow) =>
    r.value === null ? "Not disclosed" : format(r.value, f);
  const weight = (r: BarRow) => (r.emphasis ? 650 : 450);
  const aria =
    `${d.title}. ` +
    d.rows
      .map(
        r =>
          `${r.label} ${r.value === null ? "not disclosed" : format(r.value, f)}${r.note ? `, ${r.note}` : ""}`
      )
      .join("; ");

  responsive(node, (el, L) => {
    const lp = L.labelPx,
      ap = L.axisPx;
    // The label column and the value gutter are measured, not fixed, so the
    // bars get every pixel the text does not need.
    const labelW = Math.max(...d.rows.map(r => textWidth(r.label, lp, weight(r))));
    const padR = Math.ceil(widest(d.rows.map(endText), lp)) + 12;
    const inlineL = Math.ceil(labelW) + 14;
    // Labels sit left of their bars while that leaves the bars at least 45% of
    // the width. Past that (phones, long labels) each label goes on its own
    // line above its bar, and the bars take the full width.
    const stacked = L.w - inlineL - padR < L.w * 0.45;
    const padL = stacked ? 0 : inlineL;
    const rowH = stacked ? lp + 24 : Math.max(30, lp + 17);
    const barH = 14;
    const barY = stacked ? lp + 5 : (rowH - barH) / 2;
    const midY = barY + barH / 2;
    const textDy = lp * 0.35; // baseline offset that centres a label on midY
    // A reference line's label rides above the first row.
    const padT = d.reference ? ap + 10 : 4;
    const plotW = L.w - padL - padR;
    // A negative bar's value label sits left of the bar end. Stretch the low
    // end of the domain until every such label ends inside the plot, so it
    // never runs into the row-label column (inline) or off the left edge
    // (stacked). Solves x(v) >= 8 + label width for the domain's low end.
    const dom: [number, number] = [domain[0], domain[1]];
    if (neg)
      for (const r of d.rows) {
        if (r.value === null || r.value >= 0) continue;
        const need = 8 + textWidth(endText(r), lp);
        if (need >= plotW) continue;
        dom[0] = Math.min(dom[0], (r.value * plotW - need * dom[1]) / (plotW - need));
      }
    const x =
      d.scale === "log"
        ? scaleLog().domain(dom).range([0, plotW])
        : scaleLinear().domain(dom).range([0, plotW]);
    const axisY = padT + d.rows.length * rowH + 4;
    const tickY = axisY + ap + 2;
    const unitY = tickY + ap + 8;
    const unitN = d.unit ? titleLines(d.unit, L.w, ap).length : 0;
    const h = Math.ceil(
      (d.unit ? unitY + (unitN - 1) * titleLineH(ap) : tickY) + ap * 0.35
    );
    // Vertical rules (gridlines, zero, reference). With labels stacked above
    // the bars, a rule would strike through every label, so it is drawn as
    // one segment per row that starts under that row's label.
    const rules = (y1: number): [number, number][] =>
      stacked
        ? [
            ...(y1 < padT ? [[y1, padT] as [number, number]] : []),
            ...d.rows.map((_r, i): [number, number] => [
              padT + i * rowH + barY - 3,
              i === d.rows.length - 1 ? axisY : padT + (i + 1) * rowH + 1,
            ]),
          ]
        : [[y1, axisY]];
    const vrule = (xv: number, y1: number) =>
      svg
        .append("g")
        .selectAll("line")
        .data(rules(y1))
        .join("line")
        .attr("x1", xv)
        .attr("x2", xv)
        .attr("y1", s => s[0])
        .attr("y2", s => s[1]);

    const svg = sized(el, L.w, h, aria);
    const hatched = hatch(
      svg,
      `h-${Math.random().toString(36).slice(2)}`,
      t.accent
    );

    // Axis: a few hairlines, labels below, the unit under those.
    const ticks = spacedTicks(
      d.scale === "log"
        ? logTicks(domain[0], domain[1])
        : (x as ReturnType<typeof scaleLinear<number, number>>).ticks(L.compact ? 3 : 4),
      v => padL + x(v),
      v => format(v, f),
      ap
    );
    ticks.forEach(v => vrule(padL + x(v), padT).attr("stroke", t.border));
    svg
      .selectAll("text.tick")
      .data(ticks)
      .join("text")
      .attr("class", "fig-axis")
      .each(function (v) {
        const c = clampAnchor(padL + x(v), textWidth(format(v, f), ap), 0, L.w);
        select(this).attr("x", c.x).attr("text-anchor", c.anchor);
      })
      .attr("y", tickY)
      .text(v => format(v, f));
    if (d.unit) axisTitle(svg, d.unit, padL + plotW / 2, unitY, L.w, ap);

    if (d.reference) {
      const rx = padL + x(d.reference.value);
      vrule(rx, padT - 4)
        .attr("stroke", SERIES[1])
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", d.reference.band ? "3 3" : null);
      // Right of the line, or left of it when it would run off the edge.
      const rw = textWidth(d.reference.label, ap);
      const right = rx + 6 + rw <= L.w;
      svg
        .append("text")
        .attr("class", "fig-axis")
        .attr("x", right ? rx + 6 : rx - 6)
        .attr("y", ap)
        .attr("text-anchor", right ? "start" : "end")
        .style("fill", SERIES[1])
        .text(d.reference.label);
    }

    const g = svg
      .selectAll<SVGGElement, BarRow>("g.row")
      .data(d.rows)
      .join("g")
      .attr("transform", (_r, i) => `translate(0,${padT + i * rowH})`)
      .on("mousemove", (event: MouseEvent, r: BarRow) =>
        showTip(
          `<strong>${r.label}</strong><br>${r.value === null ? "Not disclosed" : format(r.value, f)}${
            r.note ? ` · ${r.note}` : ""
          }${r.detail ? `<br><span class="fig-tip-muted">${r.detail}</span>` : ""}`,
          event
        )
      )
      .on("mouseleave", hideTip);

    g.append("rect")
      .attr("width", L.w)
      .attr("height", rowH)
      .attr("fill", "transparent");
    g.append("text")
      .attr("x", stacked ? 0 : padL - 12)
      .attr("y", stacked ? lp : midY + textDy)
      .attr("text-anchor", stacked ? "start" : "end")
      .attr("class", "fig-label")
      .attr("font-weight", weight)
      .text(r => (stacked ? fitText(r.label, L.w, lp, weight(r)) : r.label));

    const x0 = d.scale === "log" ? x(domain[0]) : neg ? x(0) : 0;
    if (neg) vrule(padL + x0, padT).attr("stroke", t.muted);
    g.filter(r => r.value !== null)
      .append("rect")
      .attr("x", r => padL + Math.min(x0, x(r.value as number)))
      .attr("y", barY)
      .attr("width", r => Math.max(2, Math.abs(x(r.value as number) - x0)))
      .attr("height", barH)
      .attr("rx", 4)
      .attr("fill", r =>
        r.estimate ? hatched : r.emphasis ? t.accent : DEEMPHASIS
      )
      .attr("stroke", r => (r.estimate ? t.accent : "none"))
      .attr("stroke-width", 1);
    const below = (r: BarRow) => r.value !== null && r.value < 0;
    g.append("text")
      .attr("x", r =>
        below(r)
          ? padL + x(r.value as number) - 8
          : padL + (r.value === null ? x0 : x(r.value)) + 8
      )
      .attr("text-anchor", r => (below(r) ? "end" : null))
      .attr("y", midY + textDy)
      .attr("class", r =>
        r.value === null ? "fig-label fig-halo" : "fig-label fig-num fig-halo"
      )
      .style("fill", r => (r.value === null ? t.muted : t.fg))
      .attr("font-style", r => (r.value === null ? "italic" : "normal"))
      .text(endText);
  });

  const items: { label: string; swatch: string; hatched?: boolean }[] = [];
  if (d.rows.some(r => r.emphasis))
    items.push({
      label: d.legendLabels?.emphasis ?? "Subject of this post",
      swatch: t.accent,
    });
  if (d.rows.some(r => !r.emphasis && !r.estimate && r.value !== null))
    items.push({
      label: d.legendLabels?.other ?? "Disclosed",
      swatch: DEEMPHASIS,
    });
  if (d.rows.some(r => r.estimate))
    items.push({
      label: "Estimate, forecast or target",
      swatch: t.accent,
      hatched: true,
    });
  if (items.length > 1) legend(node, items);
}

/* --------------------------------------------------------------- columns */

type ColValue = number | [number, number] | null;

type ColumnsSpec = Base & {
  type: "columns";
  categories: string[];
  series: {
    name: string;
    values: ColValue[];
    /** Per-category: this value is guidance or forecast, drawn hatched. */
    estimate?: boolean[];
  }[];
  mode?: "grouped" | "stacked";
  format?: Format;
  unit?: string;
  /** Text above each category, e.g. an inference share. */
  annotations?: (string | null)[];
  /** Per-bar value labels; off for dense stacks where the axis carries them. */
  showValues?: boolean;
};

function columns(node: Element, d: ColumnsSpec) {
  const t = theme();
  const f = d.format ?? "number";
  const mode = d.mode ?? "grouped";
  const top = (v: ColValue) => (v === null ? 0 : Array.isArray(v) ? v[1] : v);
  const maxV =
    mode === "stacked"
      ? Math.max(
          ...d.categories.map((_c, i) =>
            d.series.reduce((a, s) => a + top(s.values[i]), 0)
          )
        )
      : Math.max(...d.series.flatMap(s => s.values.map(top)));
  const valueText = (v: Exclude<ColValue, null>) =>
    Array.isArray(v) ? `${format(v[0], f)}–${format(v[1], f)}` : format(v, f);
  const aria =
    `${d.title}. ` +
    d.categories
      .map(
        (c, i) =>
          `${c}: ` +
          d.series
            .map(s => {
              const v = s.values[i];
              return `${s.name} ${v === null ? "not available" : Array.isArray(v) ? `${format(v[0], f)} to ${format(v[1], f)}` : format(v, f)}`;
            })
            .join(", ")
      )
      .join("; ");

  responsive(node, (el, L) => {
    const lp = L.labelPx,
      ap = L.axisPx;
    const h = L.compact ? 260 : 300;
    // Headroom for the value labels over the tallest column, then rounded out
    // so the top gridline is at or above every column.
    const ax = linearAxis(0, maxV * 1.1, 4);
    const y = scaleLinear().domain(ax.domain);
    const ticks = ax.ticks;
    const padL = Math.ceil(widest(ticks.map(v => format(v, f)), ap)) + 10,
      padR = 4;
    const plotW = L.w - padL - padR;
    const x0 = scaleBand()
      .domain(d.categories)
      .range([0, plotW])
      .paddingInner(0.35)
      .paddingOuter(0.2);
    // Category names wrap onto up to three lines rather than run into each
    // other or lose their last word.
    const catLines = d.categories.map(c => wrapText(c, x0.step() - 6, lp, 3));
    const nLines = Math.max(1, ...catLines.map(l => l.length));
    // The unit is the value axis title, top left over the tick labels.
    const titleH = d.unit
      ? ap + 10 + (titleLines(d.unit, L.w, ap).length - 1) * titleLineH(ap)
      : 0;
    const annH = d.annotations?.some(Boolean) ? lp + 8 : 0;
    const padT = titleH + annH + 10;
    const padB = lp + 6 + (nLines - 1) * (lp + 3) + Math.ceil(lp * 0.35);
    const plotH = h - padT - padB;
    y.range([plotH, 0]);
    // A category never gets more than 150px, however few there are: wide slabs
    // read loud, and the value is in the height, not the area.
    const band = Math.min(x0.bandwidth(), 150);
    const inset = (x0.bandwidth() - band) / 2;
    const cx0 = (c: string) => (x0(c) ?? 0) + inset;
    const x1 = scaleBand()
      .domain(d.series.map(s => s.name))
      .range([0, band])
      .paddingInner(0.12);
    // Value labels over grouped columns only when each fits over its column;
    // on a narrow screen the axis and the tooltip carry the numbers.
    const allValues = d.series.flatMap(s =>
      s.values.filter((v): v is Exclude<ColValue, null> => v !== null)
    );
    const valuesFit =
      mode !== "grouped" ||
      widest(allValues.map(valueText), ap) <= x1.step() + 2;

    const svg = sized(el, L.w, h, aria);
    if (d.unit) axisTitle(svg, d.unit, 0, ap, L.w, ap, "start");
    const plot = svg.append("g").attr("transform", `translate(${padL},${padT})`);

    // Grid: hairlines behind the marks.
    plot
      .selectAll("line.grid")
      .data(ticks)
      .join("line")
      .attr("x1", 0)
      .attr("x2", plotW)
      .attr("y1", v => y(v))
      .attr("y2", v => y(v))
      .attr("stroke", t.border);
    plot
      .selectAll("text.grid")
      .data(ticks)
      .join("text")
      .attr("class", "fig-axis")
      .attr("x", -8)
      .attr("y", v => y(v) + ap * 0.35)
      .attr("text-anchor", "end")
      .text(v => format(v, f));
    plot
      .selectAll("text.cat")
      .data(d.categories)
      .join("text")
      .attr("class", "fig-label")
      .attr("y", plotH + lp + 6)
      .attr("text-anchor", "middle")
      .each(function (c, i) {
        setLines(this as SVGTextElement, catLines[i], cx0(c) + band / 2, lp + 3);
      });

    const patterns = d.series.map((_s, i) =>
      hatch(svg, `hc-${i}-${Math.random().toString(36).slice(2)}`, seriesColor(i))
    );

    d.categories.forEach((c, ci) => {
      let stackBase = 0;
      d.series.forEach((s, si) => {
        const v = s.values[ci];
        if (v === null) return;
        const est = s.estimate?.[ci] ?? false;
        const color = seriesColor(si);
        const lo = Array.isArray(v) ? v[0] : v;
        const hi = Array.isArray(v) ? v[1] : v;
        const bx = mode === "stacked" ? cx0(c) : cx0(c) + (x1(s.name) ?? 0);
        const bw = mode === "stacked" ? band : x1.bandwidth();
        const yTop = mode === "stacked" ? y(stackBase + hi) : y(hi);
        const yBot =
          mode === "stacked"
            ? y(stackBase + (Array.isArray(v) ? lo : 0))
            : y(Array.isArray(v) ? lo : 0);
        // In a stack, the base of the block is the top of the one below; keep a
        // 2px surface gap so adjacent fills never touch.
        const gap = mode === "stacked" && si > 0 ? 2 : 0;
        const g = plot
          .append("g")
          .on("mousemove", (event: MouseEvent) =>
            showTip(
              `<strong>${s.name}</strong> · ${c}<br>${
                Array.isArray(v)
                  ? `${format(v[0], f)} to ${format(v[1], f)}`
                  : format(v, f)
              }${est ? ` <span class="fig-tip-muted">(guidance or forecast)</span>` : ""}`,
              event
            )
          )
          .on("mouseleave", hideTip);
        g.append("rect")
          .attr("x", bx)
          .attr("y", yTop)
          .attr("width", bw)
          .attr("height", Math.max(0, yBot - yTop - gap))
          .attr("rx", mode === "stacked" && si < d.series.length - 1 ? 0 : 4)
          .attr("fill", est ? patterns[si] : color)
          .attr("stroke", est ? color : "none")
          .attr("stroke-width", 1);
        if (d.showValues === false || !valuesFit) {
          // axis and tooltip carry the values
        } else if (mode === "grouped") {
          g.append("text")
            .attr("class", "fig-axis fig-num")
            .attr("x", bx + bw / 2)
            .attr("y", yTop - 5)
            .attr("text-anchor", "middle")
            .style("fill", t.fg)
            .text(valueText(v));
        } else if (yBot - yTop > ap + 8 && textWidth(format(hi, f), ap) < bw - 4) {
          g.append("text")
            .attr("class", "fig-axis fig-num")
            .attr("x", bx + bw / 2)
            .attr("y", (yTop + yBot) / 2 + ap * 0.35)
            .attr("text-anchor", "middle")
            .style("fill", est ? t.fg : "#fff")
            .text(format(hi, f));
        }
        stackBase += hi;
      });
      const a = d.annotations?.[ci];
      if (a)
        plot
          .append("text")
          .attr("class", "fig-label")
          .attr("x", cx0(c) + band / 2)
          .attr("y", -10)
          .attr("text-anchor", "middle")
          .style("fill", t.muted)
          .text(fitText(a, x0.step(), lp));
    });
  });

  const anyEst = d.series.some(s => s.estimate?.some(Boolean));
  legend(node, [
    ...d.series.map((s, i) => ({ label: s.name, swatch: seriesColor(i) })),
    ...(anyEst
      ? [{ label: "Guidance or forecast", swatch: t.fg, hatched: true }]
      : []),
  ]);
}

/* ------------------------------------------------------------------ dots */

type DotRow = {
  label: string;
  value: number;
  /** measured: filled. claimed: a target, projection or vendor claim; hollow. */
  class: "measured" | "claimed";
  sublabel?: string;
  detail?: string;
  /** Rows sharing a pair id are joined by a bracket, e.g. two figures for one claim. */
  pair?: string;
};

type DotsSpec = Base & {
  type: "dots";
  rows: DotRow[];
  format?: Format;
  scale?: "linear" | "log";
  domain?: [number, number];
  unit?: string;
  classLabels?: { measured: string; claimed: string };
};

function dots(node: Element, d: DotsSpec) {
  const t = theme();
  const f = d.format ?? "number";
  const rowH = 34,
    padT = 12,
    padB = 34,
    padL = 236,
    padR = 40,
    w = 680;
  const plotW = w - padL - padR;
  const h = padT + d.rows.length * rowH + padB;
  const vals = d.rows.map(r => r.value);
  const [lo, hi] = d.domain ?? [Math.min(...vals), Math.max(...vals)];
  const x =
    d.scale === "log"
      ? scaleLog().domain([lo, hi]).range([0, plotW])
      : scaleLinear().domain([lo, hi]).range([0, plotW]);

  const svg = svgIn(
    node,
    w,
    h,
    `${d.title}. ` +
      d.rows
        .map(r => `${r.label} ${format(r.value, f)} (${r.class})`)
        .join("; ")
  );
  const axisY = padT + d.rows.length * rowH;
  const ticks = d.scale === "log" ? logTicks(lo, hi) : x.ticks(5);
  svg
    .selectAll("line.tick")
    .data(ticks)
    .join("line")
    .attr("x1", v => padL + x(v))
    .attr("x2", v => padL + x(v))
    .attr("y1", padT)
    .attr("y2", axisY)
    .attr("stroke", t.border);
  svg
    .selectAll("text.tick")
    .data(ticks)
    .join("text")
    .attr("class", "fig-axis")
    .attr("x", v => padL + x(v))
    .attr("y", axisY + 14)
    .attr("text-anchor", "middle")
    .text(v => format(v, f));

  // Pair brackets: rows that share an id are one claim with two values.
  const pairs = new Map<string, number[]>();
  d.rows.forEach((r, i) => {
    if (r.pair) pairs.set(r.pair, [...(pairs.get(r.pair) ?? []), i]);
  });
  for (const idx of pairs.values()) {
    if (idx.length < 2) continue;
    const y1 = padT + idx[0] * rowH + rowH / 2,
      y2 = padT + idx[idx.length - 1] * rowH + rowH / 2;
    svg
      .append("path")
      .attr("d", `M${padL - 8},${y1} h-6 V${y2} h6`)
      .attr("fill", "none")
      .attr("stroke", SERIES[1])
      .attr("stroke-width", 1.5);
  }

  const g = svg
    .selectAll<SVGGElement, DotRow>("g.row")
    .data(d.rows)
    .join("g")
    .attr("transform", (_r, i) => `translate(0,${padT + i * rowH})`)
    .on("mousemove", (event: MouseEvent, r: DotRow) =>
      showTip(
        `<strong>${r.label}</strong><br>${format(r.value, f)} · ${
          r.class === "measured"
            ? (d.classLabels?.measured ?? "measured")
            : (d.classLabels?.claimed ?? "claimed")
        }${r.detail ? `<br><span class="fig-tip-muted">${r.detail}</span>` : ""}`,
        event
      )
    )
    .on("mouseleave", hideTip);
  g.append("rect")
    .attr("width", w)
    .attr("height", rowH)
    .attr("fill", "transparent");
  g.append("text")
    .attr("x", padL - 22)
    .attr("y", rowH / 2 + 4)
    .attr("text-anchor", "end")
    .attr("class", "fig-label")
    .text(r => r.label);
  g.append("line")
    .attr("x1", padL)
    .attr("x2", r => padL + x(r.value))
    .attr("y1", rowH / 2)
    .attr("y2", rowH / 2)
    .attr("stroke", t.border);
  g.append("circle")
    .attr("cx", r => padL + x(r.value))
    .attr("cy", rowH / 2)
    .attr("r", 6)
    .attr("fill", r => (r.class === "measured" ? t.accent : t.surface))
    .attr("stroke", t.accent)
    .attr("stroke-width", 2);
  g.append("text")
    .attr("x", r => padL + x(r.value) + 12)
    .attr("y", rowH / 2 + 4)
    .attr("class", "fig-label fig-num")
    .text(r => format(r.value, f));

  legend(node, [
    { label: d.classLabels?.measured ?? "Measured", swatch: t.accent },
    {
      label: d.classLabels?.claimed ?? "Claimed, projected or target",
      swatch: t.accent,
      hatched: true,
    },
  ]);
}

/* ---------------------------------------------------------------- series */

/** A point has either a month ("2026-08") or a numeric x. */
type Point = { date?: string; x?: number; value: number; note?: string };
type Entity = { name: string; points: Point[]; emphasis?: boolean };

type SeriesSpec = Base & {
  type: "series";
  entities: Entity[];
  format?: Format;
  unit?: string;
  /** For numeric x: its format and axis label. */
  xFormat?: Format;
  xLabel?: string;
  yScale?: "linear" | "log";
  /** A dashed reference curve y = k / x, for a relation the points imply. */
  guide?: { k: number; label: string };
};

function monthIndex(date: string): number {
  const [y, m] = date.split("-").map(Number);
  return y * 12 + (m - 1);
}
function monthLabel(date: string): string {
  const [y, m] = date.split("-").map(Number);
  return `${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]} ${y}`;
}

function series(node: Element, d: SeriesSpec) {
  const t = theme();
  const f = d.format ?? "number";
  const dated = d.entities[0]?.points[0]?.date !== undefined;
  const xv = (p: Point) => (p.date ? monthIndex(p.date) : (p.x ?? 0));
  const xText = (v: number) =>
    dated
      ? monthLabel(`${Math.floor(v / 12)}-${(v % 12) + 1}`)
      : format(v, d.xFormat ?? "number");
  const all = d.entities.flatMap(e => e.points);
  const xs = all.map(xv);
  const ys = all.map(p => p.value);
  const color = (e: Entity, i: number) =>
    e.emphasis === false ? DEEMPHASIS : seriesColor(i);
  const sorted = d.entities.map(e => [...e.points].sort((a, b) => xv(a) - xv(b)));
  // Direct labels on every point when there are few; the endpoint otherwise.
  const labelled = sorted.map(pts => (pts.length <= 3 ? pts : [pts[pts.length - 1]]));
  const aria =
    `${d.title}. ` +
    d.entities
      .map(
        e =>
          `${e.name}: ` +
          e.points.map(p => `${xText(xv(p))} ${format(p.value, f)}`).join(", ")
      )
      .join("; ");

  responsive(node, (el, L) => {
    const lp = L.labelPx,
      ap = L.axisPx;
    const h = L.compact ? 250 : 300;
    const x = scaleLinear().domain([Math.min(...xs), Math.max(...xs)]);
    // Starts at 0 unless a value dips below it, so a small negative
    // measurement is drawn where it is rather than under the axis. The ends
    // are ticks, so the top gridline sits at or above every point.
    const ax = linearAxis(Math.min(0, ...ys), Math.max(...ys), L.compact ? 4 : 5);
    const logLo = 10 ** Math.floor(Math.log10(Math.min(...ys))),
      logHi = Math.max(...ys) * 1.3;
    const y =
      d.yScale === "log"
        ? scaleLog().domain([logLo, logHi])
        : scaleLinear().domain(ax.domain);
    const yTicks = d.yScale === "log" ? logTicks(logLo, logHi) : ax.ticks;
    // Gutters from the text they hold: tick labels on the left, the end
    // labels of the lines on the right.
    const padL = Math.ceil(widest(yTicks.map(v => format(v, f)), ap)) + 10;
    const padR =
      Math.ceil(
        widest(
          labelled.map(ps => format(ps[ps.length - 1].value, f)),
          lp
        )
      ) + 12;
    // The unit is the value axis title, top left; under it, room for the label
    // over the highest point.
    const titleH = d.unit
      ? ap + 10 + (titleLines(d.unit, L.w, ap).length - 1) * titleLineH(ap)
      : 0;
    const padT = titleH + lp + 10;
    const xLabelN = d.xLabel ? titleLines(d.xLabel, L.w, ap).length : 0;
    const padB =
      ap + 8 + (d.xLabel ? ap + 10 + (xLabelN - 1) * titleLineH(ap) : 0) + Math.ceil(ap * 0.35);
    const plotW = L.w - padL - padR,
      plotH = h - padT - padB;
    x.range([0, plotW]);
    y.range([plotH, 0]);

    const svg = sized(el, L.w, h, aria);
    if (d.unit) axisTitle(svg, d.unit, 0, ap, L.w, ap, "start");
    const plot = svg.append("g").attr("transform", `translate(${padL},${padT})`);
    plot
      .selectAll("line.grid")
      .data(yTicks)
      .join("line")
      .attr("x1", 0)
      .attr("x2", plotW)
      .attr("y1", v => y(v))
      .attr("y2", v => y(v))
      .attr("stroke", t.border);
    plot
      .selectAll("text.grid")
      .data(yTicks)
      .join("text")
      .attr("class", "fig-axis")
      .attr("x", -8)
      .attr("y", v => y(v) + ap * 0.35)
      .attr("text-anchor", "end")
      .text(v => format(v, f));
    // Ticks only where data exists when the axis is dates: no invented months.
    // Labels that would touch a neighbour are dropped, never stacked.
    const xTicks = spacedTicks(
      dated ? [...new Set(xs)].sort((a, b) => a - b) : x.ticks(L.compact ? 4 : 6),
      v => x(v),
      xText,
      ap
    );
    plot
      .selectAll("text.xt")
      .data(xTicks)
      .join("text")
      .attr("class", "fig-axis")
      .each(function (v) {
        const c = clampAnchor(x(v), textWidth(xText(v), ap), -padL, plotW + padR);
        select(this).attr("x", c.x).attr("text-anchor", c.anchor);
      })
      .attr("y", plotH + ap + 6)
      .text(v => xText(v));
    if (d.xLabel)
      axisTitle(svg, d.xLabel, padL + plotW / 2, padT + plotH + 2 * ap + 16, L.w, ap);

    if (d.guide) {
      const k = d.guide.k;
      const [x0, x1] = x.domain();
      const N = 80;
      const pts: string[] = [];
      for (let i = 0; i <= N; i++) {
        const gx = x0 + ((x1 - x0) * i) / N;
        if (gx <= 0) continue;
        pts.push(`${pts.length ? "L" : "M"}${x(gx)},${y(k / gx)}`);
      }
      plot
        .append("path")
        .attr("d", pts.join(" "))
        .attr("fill", "none")
        .attr("stroke", t.muted)
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "4 4");
    }

    const labels: { el: SVGTextElement; x: number; y: number }[] = [];
    // Every line in plot coordinates, to keep direct labels off the others.
    const paths = sorted.map(pts => pts.map(p => [x(xv(p)), y(p.value)] as [number, number]));
    d.entities.forEach((e, i) => {
      const c = color(e, i);
      const pts = sorted[i];
      if (pts.length > 1)
        plot
          .append("path")
          .attr(
            "d",
            pts
              .map((p, k) => `${k ? "L" : "M"}${x(xv(p))},${y(p.value)}`)
              .join(" ")
          )
          .attr("fill", "none")
          .attr("stroke", c)
          .attr("stroke-width", 2);
      plot
        .selectAll(`circle.e${i}`)
        .data(pts)
        .join("circle")
        .attr("cx", p => x(xv(p)))
        .attr("cy", p => y(p.value))
        .attr("r", L.compact ? 4 : 5)
        .attr("fill", c)
        .attr("stroke", t.surface)
        .attr("stroke-width", 2)
        .on("mousemove", (event: MouseEvent, p: Point) =>
          showTip(
            `<strong>${e.name}</strong> · ${xText(xv(p))}<br>${format(p.value, f)}${
              p.note ? `<br><span class="fig-tip-muted">${p.note}</span>` : ""
            }`,
            event
          )
        )
        .on("mouseleave", hideTip);
      plot
        .selectAll(`text.l${i}`)
        .data(labelled[i])
        .join("text")
        .attr("class", "fig-label fig-num")
        .style("fill", c)
        .text(p => format(p.value, f))
        .each(function (p) {
          // Up and right of the point by default; if another series' line
          // runs through that spot, the first free spot around the point.
          const px = x(xv(p)),
            py = y(p.value);
          const w = textWidth(format(p.value, f), lp);
          const spots: [number, number, "start" | "end" | "middle"][] = [
            [px + 8, py - 9, "start"],
            [px + 8, py + lp + 6, "start"],
            [px - 8, py - 9, "end"],
            [px - 8, py + lp + 6, "end"],
            [px, py - 12, "middle"],
            [px, py + lp + 10, "middle"],
          ];
          const left = (sx: number, a: string) =>
            a === "start" ? sx : a === "end" ? sx - w : sx - w / 2;
          const fits = ([sx, sy, a]: [number, number, string]) => {
            const x1 = left(sx, a),
              x2 = x1 + w,
              y1 = sy - lp * 0.8,
              y2 = sy + lp * 0.25;
            if (x1 < -padL || x2 > plotW + padR || y1 < -padT || y2 > plotH) return false;
            return !paths.some(
              (ps, j) => j !== i && ps.some((q, k) => k > 0 && segHitsBox(ps[k - 1], q, x1 - 2, y1 - 2, x2 + 2, y2 + 2))
            );
          };
          const [sx, sy, a] = spots.find(fits) ?? spots[0];
          select(this).attr("x", sx).attr("y", sy).attr("text-anchor", a);
          labels.push({ el: this, x: sx, y: sy });
        });
    });
    // Nudge direct labels that would print over each other (same x, near-equal
    // values) apart vertically, one label height (the font size plus leading).
    const lh = lp + 3;
    labels.sort((a, b) => a.y - b.y);
    labels.forEach((l, k) => {
      for (let j = 0; j < k; j++) {
        const o = labels[j];
        if (Math.abs(o.x - l.x) < 36 && l.y - o.y < lh) l.y = o.y + lh;
      }
      l.el.setAttribute("y", String(l.y));
    });
  });

  if (d.entities.length > 1)
    legend(
      node,
      d.entities.map((e, i) => ({ label: e.name, swatch: color(e, i) }))
    );
}

/* -------------------------------------------------------------- timeline */

type Event = { date: string; label: string; shipped: boolean; detail?: string };
type Lane = { name: string; events: Event[] };

type TimelineSpec = Base & {
  type: "timeline";
  start: string;
  end: string;
  lanes: Lane[];
  /** Legend and tooltip wording for filled vs hollow markers. */
  states?: { done: string; pending: string };
};

function timeline(node: Element, d: TimelineSpec) {
  const t = theme();
  const states = d.states ?? {
    done: "Shipped",
    pending: "Announced, not yet shipped",
  };
  const w = 680,
    laneH = 78,
    padT = 24,
    padB = 30,
    padL = 110,
    padR = 24;
  const plotW = w - padL - padR;
  const h = padT + d.lanes.length * laneH + padB;
  const x = scaleLinear()
    .domain([monthIndex(d.start), monthIndex(d.end)])
    .range([0, plotW]);

  const svg = svgIn(
    node,
    w,
    h,
    `${d.title}. ` +
      d.lanes
        .map(
          l =>
            `${l.name}: ` +
            l.events
              .map(
                e =>
                  `${monthLabel(e.date)} ${e.label}${e.shipped ? "" : " (announced)"}`
              )
              .join(", ")
        )
        .join("; ")
  );
  // Year and quarter ticks along the top.
  const months: number[] = [];
  for (let m = monthIndex(d.start); m <= monthIndex(d.end); m++)
    if (m % 3 === 0) months.push(m);
  // Quarter labels need about 60px each; past that, label years only.
  const yearsOnly = x(3) - x(0) < 60;
  const tickText = (m: number) =>
    yearsOnly
      ? m % 12 === 0
        ? String(Math.floor(m / 12))
        : ""
      : `${["Q1", "Q2", "Q3", "Q4"][(m % 12) / 3]} ${Math.floor(m / 12)}`;
  svg
    .selectAll("line.q")
    .data(months)
    .join("line")
    .attr("x1", m => padL + x(m))
    .attr("x2", m => padL + x(m))
    .attr("y1", padT - 4)
    .attr("y2", padT + d.lanes.length * laneH)
    .attr("stroke", t.border);
  svg
    .selectAll("text.q")
    .data(months)
    .join("text")
    .attr("class", "fig-axis")
    .attr("x", m => padL + x(m) + 3)
    .attr("y", padT - 8)
    .text(tickText);

  d.lanes.forEach((l, li) => {
    const y0 = padT + li * laneH;
    svg
      .append("text")
      .attr("class", "fig-label")
      .attr("x", padL - 12)
      .attr("y", y0 + laneH / 2 + 4)
      .attr("text-anchor", "end")
      .attr("font-weight", 650)
      .text(l.name);
    svg
      .append("line")
      .attr("x1", padL)
      .attr("x2", padL + plotW)
      .attr("y1", y0 + laneH / 2)
      .attr("y2", y0 + laneH / 2)
      .attr("stroke", t.border);
    const color = SERIES[li % SERIES.length];
    const evs = [...l.events].sort(
      (a, b) => monthIndex(a.date) - monthIndex(b.date)
    );
    evs.forEach((e, k) => {
      const cx = padL + x(monthIndex(e.date));
      const cy = y0 + laneH / 2;
      // Alternate label rows so neighbours don't collide.
      const above = k % 2 === 0;
      const g = svg
        .append("g")
        .on("mousemove", (event: MouseEvent) =>
          showTip(
            `<strong>${e.label}</strong><br>${monthLabel(e.date)} · ${(e.shipped ? states.done : states.pending).toLowerCase()}${
              e.detail
                ? `<br><span class="fig-tip-muted">${e.detail}</span>`
                : ""
            }`,
            event
          )
        )
        .on("mouseleave", hideTip);
      g.append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", 6)
        .attr("fill", e.shipped ? color : t.surface)
        .attr("stroke", color)
        .attr("stroke-width", 2);
      g.append("line")
        .attr("x1", cx)
        .attr("x2", cx)
        .attr("y1", cy + (above ? -8 : 8))
        .attr("y2", cy + (above ? -18 : 18))
        .attr("stroke", t.border);
      const half = (e.label.length * 6) / 2;
      const anchor =
        cx - half < padL ? "start" : cx + half > w - padR ? "end" : "middle";
      g.append("text")
        .attr("class", "fig-axis")
        .attr("x", anchor === "start" ? padL : anchor === "end" ? w - padR : cx)
        .attr("y", cy + (above ? -22 : 30))
        .attr("text-anchor", anchor)
        .style("fill", t.fg)
        .text(e.label);
    });
  });
  // Only show the hollow-marker swatch when some event actually uses it.
  const anyPending = d.lanes.some(l => l.events.some(e => !e.shipped));
  legend(node, [
    { label: states.done, swatch: t.accent },
    ...(anyPending
      ? [{ label: states.pending, swatch: t.accent, hatched: true }]
      : []),
  ]);
}

/* ---------------------------------------------------------------- matrix */

type MatrixSpec = Base & {
  type: "matrix";
  columns: string[];
  rows: {
    label: string;
    cells: (string | null)[];
    emphasis?: boolean;
    note?: string;
  }[];
};

function matrix(node: Element, d: MatrixSpec) {
  const t = theme();
  const w = 680,
    rowH = 34,
    headH = 30,
    padL = 172;
  const colW = (w - padL) / d.columns.length;
  const h = headH + d.rows.length * rowH + 8;
  const svg = svgIn(
    node,
    w,
    h,
    `${d.title}. ` +
      d.rows
        .map(
          r =>
            `${r.label}: ` +
            r.cells
              .map((c, i) => `${d.columns[i]} ${c ?? "not published"}`)
              .join(", ")
        )
        .join("; ")
  );
  svg
    .selectAll("text.col")
    .data(d.columns)
    .join("text")
    .attr("class", "fig-axis")
    .attr("x", (_c, i) => padL + i * colW + colW / 2)
    .attr("y", 18)
    .attr("text-anchor", "middle")
    .style("fill", t.fg)
    .text(c => c);
  d.rows.forEach((r, ri) => {
    const y0 = headH + ri * rowH;
    svg
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", y0)
      .attr("y2", y0)
      .attr("stroke", t.border);
    svg
      .append("text")
      .attr("class", "fig-label")
      .attr("x", padL - 12)
      .attr("y", y0 + rowH / 2 + 4)
      .attr("text-anchor", "end")
      .attr("font-weight", r.emphasis ? 650 : 450)
      .text(r.label);
    r.cells.forEach((c, ci) => {
      const cx = padL + ci * colW + colW / 2;
      const g = svg
        .append("g")
        .on("mousemove", (event: MouseEvent) =>
          showTip(
            `<strong>${r.label}</strong> · ${d.columns[ci]}<br>${c ?? "Not published"}${
              r.note ? `<br><span class="fig-tip-muted">${r.note}</span>` : ""
            }`,
            event
          )
        )
        .on("mouseleave", hideTip);
      g.append("rect")
        .attr("x", padL + ci * colW + 2)
        .attr("y", y0 + 4)
        .attr("width", colW - 4)
        .attr("height", rowH - 8)
        .attr("rx", 4)
        .attr("fill", c ? `${t.accent}14` : "transparent")
        .attr("stroke", c ? "none" : t.border)
        .attr("stroke-dasharray", c ? null : "3 3");
      g.append("text")
        .attr("class", "fig-axis")
        .attr("x", cx)
        .attr("y", y0 + rowH / 2 + 4)
        .attr("text-anchor", "middle")
        .style("fill", c ? t.fg : t.muted)
        .attr("font-style", c ? "normal" : "italic")
        .text(c ?? "Not published");
    });
  });
}

/* ----------------------------------------------------------------- curve */

/**
 * A model, not a measurement. The memo's argument is that a service-robot
 * business flips sign somewhere between one and ten robots per operator, and
 * that no public number places the company on that axis. So the chart shows
 * the curve and deliberately draws no marker on it.
 */
type CurveSpec = Base & {
  type: "curve";
  price: number;
  scenarios: { name: string; operatorCost: number }[];
  service: {
    label: string;
    min: number;
    max: number;
    default: number;
    step: number;
  };
  ratio: { min: number; max: number };
};

function curve(node: Element, d: CurveSpec) {
  const t = theme();
  const w = 680,
    h = 300,
    padT = 20,
    padB = 44,
    padL = 70,
    padR = 150;
  const plotW = w - padL - padR,
    plotH = h - padT - padB;

  const controls = document.createElement("div");
  controls.className = "fig-controls";
  const lab = document.createElement("label");
  lab.textContent = d.service.label;
  const input = document.createElement("input");
  input.type = "range";
  input.min = String(d.service.min);
  input.max = String(d.service.max);
  input.step = String(d.service.step);
  input.value = String(d.service.default);
  const val = document.createElement("span");
  val.className = "fig-value";
  controls.append(lab, input, val);
  node.appendChild(controls);

  const svg = svgIn(node, w, h, `${d.title}. A sensitivity model, not data.`);
  const plot = svg.append("g").attr("transform", `translate(${padL},${padT})`);
  const x = scaleLinear().domain([d.ratio.min, d.ratio.max]).range([0, plotW]);
  const margin = (ratio: number, op: number, svc: number) =>
    d.price - op / ratio - svc;
  const yMin = Math.min(
    ...d.scenarios.map(s => margin(d.ratio.min, s.operatorCost, d.service.max))
  );
  const y = scaleLinear().domain([yMin, d.price]).range([plotH, 0]).nice();

  const yt = y.ticks(5);
  plot
    .selectAll("line.grid")
    .data(yt)
    .join("line")
    .attr("x1", 0)
    .attr("x2", plotW)
    .attr("y1", v => y(v))
    .attr("y2", v => y(v))
    .attr("stroke", v => (v === 0 ? t.muted : t.border))
    .attr("stroke-width", v => (v === 0 ? 1.5 : 1));
  plot
    .selectAll("text.grid")
    .data(yt)
    .join("text")
    .attr("class", "fig-axis")
    .attr("x", -8)
    .attr("y", v => y(v) + 3)
    .attr("text-anchor", "end")
    .text(v => money(v));
  const xt = x.ticks(d.ratio.max - d.ratio.min);
  plot
    .selectAll("text.x")
    .data(xt)
    .join("text")
    .attr("class", "fig-axis")
    .attr("x", v => x(v))
    .attr("y", plotH + 18)
    .attr("text-anchor", "middle")
    .text(v => String(v));
  plot
    .append("text")
    .attr("class", "fig-axis")
    .attr("x", plotW / 2)
    .attr("y", plotH + 34)
    .attr("text-anchor", "middle")
    .text("Robots per operator");

  const paths = d.scenarios.map((_s, i) =>
    plot
      .append("path")
      .attr("fill", "none")
      .attr("stroke", SERIES[i % SERIES.length])
      .attr("stroke-width", 2)
  );
  const cross = d.scenarios.map((_s, i) =>
    plot
      .append("circle")
      .attr("r", 4)
      .attr("fill", t.surface)
      .attr("stroke", SERIES[i % SERIES.length])
      .attr("stroke-width", 2)
  );

  function draw(svc: number) {
    val.textContent = `${money(svc)} per robot-year`;
    const N = 60;
    d.scenarios.forEach((s, i) => {
      const pts: [number, number][] = [];
      for (let k = 0; k <= N; k++) {
        const r = d.ratio.min + ((d.ratio.max - d.ratio.min) * k) / N;
        pts.push([x(r), y(margin(r, s.operatorCost, svc))]);
      }
      paths[i].attr(
        "d",
        pts.map((p, k) => `${k ? "L" : "M"}${p[0]},${p[1]}`).join(" ")
      );
      // Where this scenario crosses zero.
      const be = s.operatorCost / (d.price - svc);
      const visible = be >= d.ratio.min && be <= d.ratio.max;
      cross[i]
        .attr("cx", x(Math.min(Math.max(be, d.ratio.min), d.ratio.max)))
        .attr("cy", y(0))
        .attr("opacity", visible ? 1 : 0);
    });
  }
  input.addEventListener("input", () => draw(Number(input.value)));
  draw(d.service.default);

  legend(
    node,
    d.scenarios.map((s, i) => ({
      label: `${s.name} (${money(s.operatorCost)} per operator)`,
      swatch: SERIES[i % SERIES.length],
    }))
  );
}

/* -------------------------------------------------------------- dispatch */

type Spec =
  | BarsSpec
  | ColumnsSpec
  | DotsSpec
  | SeriesSpec
  | TimelineSpec
  | MatrixSpec
  | CurveSpec;

export async function ventureFigure(node: Element, name: string) {
  return specFigure(node, `ventures/${name}`);
}

/**
 * The same JSON chart forms for any post: `chart:tiny-circuits/head-patching`
 * renders /data/tiny-circuits/head-patching.json.
 */
export async function specFigure(node: Element, path: string) {
  const spec = await json<Spec>(`data/${path}.json`);
  node.classList.add("fig-venture");
  heading(node, spec);
  switch (spec.type) {
    case "bars":
      bars(node, spec);
      break;
    case "columns":
      columns(node, spec);
      break;
    case "dots":
      dots(node, spec);
      break;
    case "series":
      series(node, spec);
      break;
    case "timeline":
      timeline(node, spec);
      break;
    case "matrix":
      matrix(node, spec);
      break;
    case "curve":
      curve(node, spec);
      break;
  }
  footer(node, spec);
}
