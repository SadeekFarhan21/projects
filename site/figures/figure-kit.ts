/**
 * Shared plumbing for the D3 figures.
 *
 * Both figure modules draw from this: the technical posts (figures.ts) and the
 * venture memos (venture-figures.ts). Theme colors are read from the CSS tokens
 * at runtime, so a token edit in src/styles/theme.css flows through without
 * touching any chart code.
 */

const token = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
  fallback;

export type Theme = {
  fg: string;
  muted: string;
  border: string;
  accent: string;
  surface: string;
};

export const theme = (): Theme => ({
  fg: token("--foreground", "#282728"),
  muted: token("--muted-foreground", "#6b7280"),
  border: token("--border", "#ece9e9"),
  accent: token("--accent", "#006cac"),
  surface: token("--background", "#fdfdfd"),
});

/**
 * Categorical slots, assigned in fixed order and never cycled. Validated on the
 * site's light surface (#fdfdfd) against the lightness band, chroma floor,
 * colorblind separation, normal-vision floor and 3:1 contrast. Slot 1 is the
 * site accent. A fourth series folds into "Other" or becomes small multiples
 * rather than getting a generated hue.
 *
 * DEEMPHASIS is not a slot: it is the gray that context marks wear in an
 * emphasis chart, where one entity is the subject and the rest are backdrop.
 */
export const SERIES = ["#006cac", "#c0572a", "#7d5ba6"] as const;
export const DEEMPHASIS = "#9aa1ab";

/**
 * Color of the i-th series. A fourth series is the "Other" slot: it takes the
 * deemphasis gray rather than wrapping round to slot 1, which drew two series
 * of one chart in the same blue.
 */
export const seriesColor = (i: number): string => SERIES[i] ?? DEEMPHASIS;

export const fmt = new Intl.NumberFormat("en-US");

/** One shared tooltip, positioned against the page. */
let tip: HTMLDivElement | null = null;
function tooltip(): HTMLDivElement {
  if (tip) return tip;
  tip = document.createElement("div");
  tip.className = "fig-tip";
  tip.setAttribute("role", "status");
  document.body.appendChild(tip);
  return tip;
}

/**
 * Hover focus. Every chart reports hovers through showTip, and the element that
 * carries the listener is the mark (or the group for one item). Keep it solid
 * and fade its same-kind siblings, so the reader sees which mark the tooltip
 * describes. Axes and labels are different element kinds and stay untouched.
 */
let hot: Element | null = null;
function focusMark(el: EventTarget | null) {
  if (!(el instanceof SVGElement) || el === hot) return;
  blurMark();
  const parent = el.parentElement;
  if (!parent) return;
  const kin = [...parent.children].filter(c => c !== el && c.tagName === el.tagName);
  if (kin.length === 0) return;
  hot = el;
  el.classList.add("fig-hot");
  kin.forEach(k => k.classList.add("fig-dim"));
}
function blurMark() {
  if (!hot) return;
  hot.parentElement
    ?.querySelectorAll(":scope > .fig-dim")
    .forEach(k => k.classList.remove("fig-dim"));
  hot.classList.remove("fig-hot");
  hot = null;
}

// Touch has no mouseleave: a tap outside any figure dismisses the tooltip.
document.addEventListener("pointerdown", e => {
  if (!(e.target instanceof Element) || !e.target.closest(".fig")) hideTip(true);
});

/**
 * Tooltip state shared with interact.ts: the element that last opened the
 * tooltip (so interact.ts can discover each chart's hoverable marks), and
 * whether the reader pinned it with a click.
 */
export const tipState = { seq: 0, target: null as Element | null, pinned: false };
export function showTip(html: string, event: MouseEvent) {
  if (tipState.pinned && event.currentTarget !== tipState.target) return;
  tipState.seq++;
  tipState.target = event.currentTarget instanceof Element ? event.currentTarget : null;
  focusMark(event.currentTarget);
  const t = tooltip();
  t.classList.toggle("is-pinned", tipState.pinned);
  t.innerHTML = html;
  t.style.opacity = "1";
  const pad = 12,
    edge = 8,
    vw = document.documentElement.clientWidth;
  // Measure at the page's left edge: an absolutely positioned box shrinks to
  // the room left of its old position, so measuring in place gives a width
  // that changes once the box moves. Never wider than the viewport.
  t.style.maxWidth = `${Math.min(352, vw - 2 * edge)}px`;
  t.style.left = "0px";
  const rect = t.getBoundingClientRect();
  let x = event.clientX + pad;
  if (x + rect.width > vw - edge) x = event.clientX - rect.width - pad;
  // On a narrow screen neither side of the finger may fit: keep it on screen.
  x = Math.max(edge, Math.min(x, vw - edge - rect.width));
  let y = event.clientY - rect.height - pad;
  if (y < edge) y = event.clientY + pad;
  t.style.left = `${x + window.scrollX}px`;
  t.style.top = `${y + window.scrollY}px`;
}

export function hideTip(force: unknown = false) {
  if (tipState.pinned && force !== true) return;
  tipState.pinned = false;
  if (tip) {
    tip.style.opacity = "0";
    tip.classList.remove("is-pinned");
  }
  blurMark();
}

/** Keep the current tooltip open until the reader clicks elsewhere. */
export function pinTip(on: boolean) {
  tipState.pinned = on && !!tipState.target;
  tip?.classList.toggle("is-pinned", tipState.pinned);
}

/**
 * The footer under a chart: the source line and the hover hint share one row
 * below a hairline, so they read as metadata rather than more caption.
 */
function footRow(node: Element): HTMLElement {
  let f = node.querySelector<HTMLElement>(":scope > .fig-foot");
  if (!f) {
    f = document.createElement("div");
    f.className = "fig-foot";
    node.appendChild(f);
  }
  return f;
}

/**
 * A one-line cue under charts that answer hover, since nothing else says so.
 * Touch screens have no hover or arrow keys, so there it would only be clutter
 * (a tap on a mark still opens its tooltip).
 */
export function hoverHint(node: Element) {
  if (node.querySelector(".fig-controls, .fig-hint")) return;
  if (window.matchMedia("(hover: none)").matches) return;
  const p = document.createElement("p");
  p.className = "fig-hint";
  p.textContent = "Hover for exact values · click to pin · arrow keys step through";
  footRow(node).appendChild(p);
}

/** Title and optional subtitle above the chart, shared by every figure. */
export function heading(node: Element, spec: { title: string; subtitle?: string }) {
  const h = document.createElement("div");
  h.className = "fig-head";
  const t = document.createElement("div");
  t.className = "fig-title";
  t.textContent = spec.title;
  h.appendChild(t);
  if (spec.subtitle) {
    const s = document.createElement("div");
    s.className = "fig-subtitle";
    s.textContent = spec.subtitle;
    h.appendChild(s);
  }
  node.appendChild(h);
}

/** A legend row of swatches under the chart. */
export function legend(
  node: Element,
  items: { label: string; swatch: string; hatched?: boolean }[]
) {
  const l = document.createElement("div");
  l.className = "fig-legend";
  for (const it of items) {
    const s = document.createElement("span");
    s.className = "fig-legend-item";
    const sw = document.createElement("i");
    sw.className = "fig-swatch";
    sw.style.background = it.hatched
      ? `repeating-linear-gradient(45deg, ${it.swatch} 0 1.5px, transparent 1.5px 4.5px)`
      : it.swatch;
    if (it.hatched) sw.style.borderColor = it.swatch;
    s.appendChild(sw);
    s.appendChild(document.createTextNode(it.label));
    l.appendChild(s);
  }
  node.appendChild(l);
}

export function caption(node: Element, text: string) {
  const c = document.createElement("figcaption");
  c.textContent = text;
  node.appendChild(c);
}

/**
 * A figure in a memo is only as good as its provenance, so the source line is a
 * separate element rather than part of the caption prose. The data files write
 * it as a lowercase phrase ("the project's peeking simulations") to follow the
 * "Source: " prefix from figures.css; it starts a sentence there, so its first
 * letter is capitalised here (CSS ::first-letter would hit the prefix instead).
 */
export function sourceLine(node: Element, text: string) {
  const s = document.createElement("p");
  s.className = "fig-source";
  s.textContent = text.charAt(0).toUpperCase() + text.slice(1);
  footRow(node).appendChild(s);
}

/* ------------------------------------------------------------ responsive */

/**
 * Charts are drawn in CSS pixels at the width their figure actually has, so a
 * 12px label is 12px on a phone too, instead of a 680-wide drawing scaled down
 * to 45% (4.5px text). Below COMPACT a chart switches to its narrow layout
 * (labels above bars, fewer ticks) and to the larger compact text sizes set by
 * `svg.fig-compact` in figures.css. Under MIN_W it scales down a little rather
 * than squeezing further.
 */
export const MIN_W = 280;
export const COMPACT = 520;

export type Layout = {
  /** Drawing width in CSS px; the viewBox is this wide. */
  w: number;
  /** Narrow layout: phones and other small containers. */
  compact: boolean;
  /** Font sizes of .fig-axis and .fig-label at this layout, for measuring. */
  axisPx: number;
  labelPx: number;
};

/** Content width of a figure: its box minus padding. 0 while not laid out. */
export function contentWidth(node: Element): number {
  const el = node as HTMLElement;
  const cs = getComputedStyle(el);
  const w =
    el.clientWidth - parseFloat(cs.paddingLeft || "0") - parseFloat(cs.paddingRight || "0");
  return w > 0 ? Math.floor(w) : 0;
}

let measureCtx: CanvasRenderingContext2D | null | undefined;
let measureFamily = "";

/** Rendered width of a label, measured with the page's font. */
export function textWidth(s: string, px: number, weight: number | string = 400): number {
  if (measureCtx === undefined) measureCtx = document.createElement("canvas").getContext("2d");
  if (!measureCtx) return s.length * px * 0.6;
  if (!measureFamily) measureFamily = getComputedStyle(document.body).fontFamily || "sans-serif";
  measureCtx.font = `${weight} ${px}px ${measureFamily}`;
  // A little slack: the webfont may still be loading when this runs.
  return measureCtx.measureText(s).width * 1.04;
}

/** The label cut to fit maxW, ending in an ellipsis when it had to be cut. */
export function fitText(s: string, maxW: number, px: number, weight: number | string = 400): string {
  if (textWidth(s, px, weight) <= maxW) return s;
  let lo = 0,
    hi = s.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(s.slice(0, mid).trimEnd() + "…", px, weight) <= maxW) lo = mid;
    else hi = mid - 1;
  }
  return lo === 0 ? "…" : s.slice(0, lo).trimEnd() + "…";
}

/** Word-wrap a label into at most maxLines lines of maxW; the last is cut. */
export function wrapText(
  s: string,
  maxW: number,
  px: number,
  maxLines = 2,
  weight: number | string = 400
): string[] {
  const words = s.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let cur = "";
  for (let i = 0; i < words.length; i++) {
    const next = cur ? `${cur} ${words[i]}` : words[i];
    if (!cur || textWidth(next, px, weight) <= maxW) {
      cur = next;
      continue;
    }
    lines.push(cur);
    cur = words[i];
    if (lines.length === maxLines - 1) {
      cur = words.slice(i).join(" ");
      break;
    }
  }
  if (cur) lines.push(cur);
  return lines.map(l => fitText(l, maxW, px, weight));
}

/** Write lines into an SVG <text> as tspans, lineH apart. */
export function setLines(text: SVGTextElement, lines: string[], x: number, lineH: number) {
  text.textContent = "";
  lines.forEach((l, i) => {
    const ts = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    ts.setAttribute("x", String(x));
    if (i) ts.setAttribute("dy", String(lineH));
    ts.textContent = l;
    text.appendChild(ts);
  });
}

/**
 * Keep tick labels that do not collide: walk left to right and drop a tick
 * whose label would touch the last one kept. The ends of the axis win over
 * the middle when only two fit.
 */
export function spacedTicks<T>(
  ticks: T[],
  pos: (v: T) => number,
  label: (v: T) => string,
  px: number,
  gap = 10
): T[] {
  const kept: T[] = [];
  let lastRight = -Infinity;
  for (const v of ticks) {
    const half = textWidth(label(v), px) / 2;
    const p = pos(v);
    if (p - half >= lastRight + gap) {
      kept.push(v);
      lastRight = p + half;
    }
  }
  return kept;
}

/** x and text-anchor for a label centred at px that must stay inside [lo, hi]. */
export function clampAnchor(px: number, tw: number, lo: number, hi: number) {
  if (px - tw / 2 < lo) return { x: lo, anchor: "start" };
  if (px + tw / 2 > hi) return { x: hi, anchor: "end" };
  return { x: px, anchor: "middle" };
}

/**
 * Draw a chart at its figure's width and draw it again when that width
 * changes (debounced, via ResizeObserver). `draw` gets a fresh, empty <svg>
 * already in the chart's place in the figure; on a redraw it replaces the old
 * one, so the HTML chrome around it (heading, controls, legend, caption) and
 * the listeners on the figure stay as they are. The figure then receives a
 * "fig:redraw" event, which interact.ts and motion.ts use to restore legend
 * isolation and to finish an entrance animation the redraw cut short.
 */
export function responsive(node: Element, draw: (svg: SVGSVGElement, L: Layout) => void) {
  const host = node as HTMLElement;
  let svg: SVGSVGElement | null = null;
  let drawnAt = -1;
  const render = (force = false) => {
    const avail = contentWidth(host);
    // Not laid out (display: none): keep what is there, or draw at desktop size.
    if (!avail && svg) return;
    const w = Math.max(MIN_W, avail || 680);
    if (w === drawnAt && !force) return;
    drawnAt = w;
    const compact = w < COMPACT;
    const next = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    next.setAttribute("width", "100%");
    if (compact) next.classList.add("fig-compact");
    const old = svg;
    const animating =
      !!old?.getAnimations?.({ subtree: true }).some(a => a.playState === "running");
    if (old) {
      // A pinned or open tooltip belongs to marks that are about to go.
      if (tipState.target && old.contains(tipState.target)) hideTip(true);
      old.replaceWith(next);
    } else host.appendChild(next);
    svg = next;
    // Keep in step with .fig-axis / .fig-label (and their svg.fig-compact
    // variants) in figures.css: layouts measure labels at these sizes.
    draw(next, { w, compact, axisPx: compact ? 11 : 12, labelPx: compact ? 12 : 13 });
    if (old) host.dispatchEvent(new CustomEvent("fig:redraw", { detail: { animating } }));
  };
  render();
  // Labels were measured with the fallback font if the webfont was still
  // loading; lay out once more when it arrives.
  if (document.fonts && document.fonts.status !== "loaded")
    document.fonts.ready.then(() => {
      measureFamily = "";
      render(true);
    });
  if (typeof ResizeObserver !== "undefined") {
    let timer = 0;
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = window.setTimeout(() => render(), 120);
    }).observe(host);
  }
}

export async function json<T>(path: string): Promise<T> {
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");
  const res = await fetch(`${base}${path.replace(/^\//, "")}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}
