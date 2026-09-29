/**
 * Entrance motion for every figure, applied after a chart has rendered.
 *
 * The chart renderers draw plain SVG; this module reads that SVG and animates
 * it by shape, so every chart form gets motion without per-chart code:
 *
 *   horizontal bars  grow from their baseline (left, or right for negatives)
 *   columns          grow up from their baseline
 *   grid cells       ripple in along the diagonal (heatmaps, matrices)
 *   lines            draw left to right; points pop in as the line reaches them
 *   areas            fade in behind the line
 *   value labels     count up to their number while the bars grow
 *   axes and text    fade in
 *
 * Motion starts when the figure scrolls into view and can be replayed.
 * Nothing moves when the reader prefers reduced motion.
 */

const EASE = "cubic-bezier(0.22, 1, 0.36, 1)";
const GROW_MS = 750;
const DRAW_MS = 1100;

const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

type Box = { el: SVGGraphicsElement; x: number; y: number; w: number; h: number };

function box(el: SVGGraphicsElement): Box | null {
  try {
    const b = el.getBBox();
    if (!(b.width > 0.5 && b.height > 0.5)) return null;
    return { el, x: b.x, y: b.y, w: b.width, h: b.height };
  } catch {
    return null;
  }
}

function visibleFill(el: Element): boolean {
  const f = (el.getAttribute("fill") ?? getComputedStyle(el).fill ?? "").trim();
  if (!f || f === "none" || f === "transparent") return false;
  const op = parseFloat(el.getAttribute("fill-opacity") ?? getComputedStyle(el).fillOpacity ?? "1");
  return op > 0.02;
}

const inDefs = (el: Element) => !!el.closest("defs, pattern, clipPath, mask");

/** Most common value (rounded), and how many elements share it. */
function mode(values: number[]): [number, number] {
  const counts = new Map<number, number>();
  for (const v of values) {
    const k = Math.round(v * 2) / 2;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  let best = NaN,
    n = 0;
  counts.forEach((c, k) => {
    if (c > n) [best, n] = [k, c];
  });
  return [best, n];
}

const near = (a: number, b: number, tol = 1) => Math.abs(a - b) <= tol;

function prep(el: Element, origin: string) {
  const s = (el as SVGElement).style;
  s.transformBox = "fill-box";
  s.transformOrigin = origin;
}

/* ------------------------------------------------------------ count-up */

const NUM = /-?\d[\d,]*(?:\.\d+)?/;

function countUp(text: SVGTextElement, delay: number, ms: number) {
  const original = text.textContent ?? "";
  const m = original.match(NUM);
  if (!m || m.index === undefined) return;
  const raw = m[0];
  const target = parseFloat(raw.replace(/,/g, ""));
  if (!isFinite(target) || target === 0) return;
  const decimals = (raw.split(".")[1] ?? "").length;
  const grouped = raw.includes(",");
  const before = original.slice(0, m.index);
  const after = original.slice(m.index + raw.length);
  const fmt = (v: number) => {
    const s = grouped
      ? v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
      : v.toFixed(decimals);
    return before + s + after;
  };
  const start = performance.now() + delay;
  text.textContent = fmt(0);
  const tick = (now: number) => {
    const t = Math.min(1, Math.max(0, (now - start) / ms));
    const e = 1 - Math.pow(1 - t, 3);
    text.textContent = t >= 1 ? original : fmt(target * e);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ------------------------------------------------------------- animate */

function play(svg: SVGSVGElement) {
  const all = <T extends Element>(sel: string) =>
    [...svg.querySelectorAll<T>(sel)].filter(e => !inDefs(e));

  // Shapes the reader hovers; cancel any earlier run first so replay is clean.
  svg.querySelectorAll("*").forEach(e => e.getAnimations?.().forEach(a => a.cancel()));

  const rects = all<SVGRectElement>("rect")
    .filter(visibleFill)
    .map(box)
    .filter((b): b is Box => !!b);

  // Background panels (anything covering most of the plot) are not marks.
  const vb = svg.viewBox.baseVal;
  const W = vb && vb.width ? vb.width : svg.getBoundingClientRect().width;
  const H = vb && vb.height ? vb.height : svg.getBoundingClientRect().height;
  const marks = rects.filter(b => !(b.w > W * 0.8 && b.h > H * 0.5));

  const animated = new Set<Element>();
  let barEnd = 0;

  // Grid cells: many rects of one size.
  const [cw, nw] = mode(marks.map(b => b.w));
  const [ch, nh] = mode(marks.map(b => b.h));
  const cells = marks.filter(b => near(b.w, cw) && near(b.h, ch));
  if (cells.length >= 12 && nw >= 12 && nh >= 12 && cells.length >= marks.length * 0.7) {
    const minX = Math.min(...cells.map(b => b.x));
    const minY = Math.min(...cells.map(b => b.y));
    cells.forEach(b => {
      const d = ((b.x - minX) / cw + (b.y - minY) / ch) * 18;
      prep(b.el, "center");
      b.el.animate(
        [
          { opacity: 0, transform: "scale(0.4)" },
          { opacity: 1, transform: "scale(1)" },
        ],
        { duration: 420, delay: Math.min(d, 900), easing: EASE, fill: "backwards" }
      );
      animated.add(b.el);
    });
    barEnd = 900;
  } else {
    // Horizontal bars share a left (or right) baseline; columns share a bottom.
    const [left, nLeft] = mode(marks.map(b => b.x));
    const [right] = mode(marks.map(b => b.x + b.w));
    const [, nBottom] = mode(marks.map(b => b.y + b.h));
    const horizontal = nLeft >= nBottom;
    // Horizontal: keep bars anchored on the shared left or right edge.
    // Vertical: every mark grows from its own bottom (stacks and ranges included).
    const bars = horizontal
      ? marks.filter(b => near(b.x, left) || near(b.x + b.w, left) || near(b.x + b.w, right))
      : marks;
    const order = [...bars].sort((a, b) => (horizontal ? a.y - b.y : a.x - b.x));
    order.forEach((b, i) => {
      const negative = horizontal ? near(b.x + b.w, left) && !near(b.x, left) : false;
      prep(b.el, horizontal ? (negative ? "right center" : "left center") : "center bottom");
      const delay = Math.min(i * 55, 700);
      b.el.animate(
        horizontal
          ? [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }]
          : [{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }],
        { duration: GROW_MS, delay, easing: EASE, fill: "backwards" }
      );
      animated.add(b.el);
      barEnd = Math.max(barEnd, delay + GROW_MS);
    });
  }

  // Lines draw themselves; filled areas fade in behind them.
  let lineSpan: { x0: number; x1: number } | null = null;
  all<SVGPathElement>("path").forEach(p => {
    const b = box(p);
    const stroke = p.getAttribute("stroke") ?? "";
    const filled = visibleFill(p);
    if (!filled && stroke && stroke !== "none" && !p.getAttribute("stroke-dasharray")) {
      let len = 0;
      try {
        len = p.getTotalLength();
      } catch {
        /* ignore */
      }
      if (len > 4) {
        p.style.strokeDasharray = `${len}`;
        const a = p.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], {
          duration: DRAW_MS,
          delay: 150,
          easing: "cubic-bezier(0.45, 0, 0.2, 1)",
          fill: "backwards",
        });
        a.onfinish = a.oncancel = () => (p.style.strokeDasharray = "");
        if (b) lineSpan = lineSpan
          ? { x0: Math.min(lineSpan.x0, b.x), x1: Math.max(lineSpan.x1, b.x + b.w) }
          : { x0: b.x, x1: b.x + b.w };
        animated.add(p);
        return;
      }
    }
    if (filled || stroke) {
      p.animate([{ opacity: 0 }, { opacity: 1 }], {
        duration: 600,
        delay: filled ? 350 : 200,
        easing: "ease-out",
        fill: "backwards",
      });
      animated.add(p);
    }
  });

  // Points pop in; on a line chart they follow the pen from left to right.
  all<SVGCircleElement>("circle")
    .map(c => ({ c, b: box(c) }))
    .filter(x => x.b)
    .sort((a, b) => a.b!.x - b.b!.x)
    .forEach(({ c, b }, i) => {
      const along = lineSpan
        ? 150 + ((b!.x - lineSpan.x0) / Math.max(1, lineSpan.x1 - lineSpan.x0)) * DRAW_MS
        : 120 + Math.min(i * 40, 900);
      prep(c, "center");
      c.animate(
        [
          { transform: "scale(0)", opacity: 0 },
          { transform: "scale(1.35)", opacity: 1, offset: 0.7 },
          { transform: "scale(1)", opacity: 1 },
        ],
        { duration: 480, delay: along, easing: "ease-out", fill: "backwards" }
      );
      animated.add(c);
    });

  // Value labels count up with their bars; everything else fades in.
  const values = all<SVGTextElement>("text").filter(
    t => !t.classList.contains("fig-axis") && NUM.test(t.textContent ?? "") && /^\s*[-$+~≈]?\s*\d/.test(t.textContent ?? "")
  );
  const valueDelay = (i: number) => Math.min(i * 55, 700);
  values.forEach((t, i) => countUp(t, valueDelay(i), GROW_MS));
  all<SVGElement>("text, line, polyline, polygon, ellipse").forEach(el => {
    if (animated.has(el)) return;
    el.animate([{ opacity: 0 }, { opacity: 1 }], {
      duration: 500,
      delay: el.tagName === "text" && !el.classList.contains("fig-axis") ? Math.min(barEnd * 0.5, 450) : 80,
      easing: "ease-out",
      fill: "backwards",
    });
  });
}

/* ---------------------------------------------------------------- wire */

const seen = new WeakSet<Element>();
let io: IntersectionObserver | null = null;

export function animateIn(node: HTMLElement) {
  const svg = node.querySelector<SVGSVGElement>(":scope > svg, svg");
  if (!svg || reduced()) return;

  replayButton(node);

  if (!("IntersectionObserver" in window)) return play(svg);
  io ??= new IntersectionObserver(
    entries =>
      entries.forEach(e => {
        if (!e.isIntersecting || seen.has(e.target)) return;
        seen.add(e.target);
        io!.unobserve(e.target);
        const s = e.target.querySelector<SVGSVGElement>("svg");
        if (s) play(s);
      }),
    { threshold: 0.25 }
  );
  io.observe(node);
}

function replayButton(node: HTMLElement) {
  if (node.querySelector(":scope > .fig-replay")) return;
  const b = document.createElement("button");
  b.type = "button";
  b.className = "fig-replay";
  b.setAttribute("aria-label", "Replay chart animation");
  b.title = "Replay";
  b.innerHTML =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 3 3 9 9 9"/></svg>';
  b.addEventListener("click", e => {
    e.stopPropagation();
    const svg = node.querySelector<SVGSVGElement>("svg");
    if (svg && !reduced()) play(svg);
  });
  node.appendChild(b);
}
