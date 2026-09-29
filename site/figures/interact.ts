/**
 * Reader interaction shared by every figure, layered on top of the chart
 * renderers' own hover tooltips (figure-kit.ts showTip/hideTip):
 *
 *   click a mark        pin its tooltip; click again or elsewhere to release
 *   arrow keys          step through the marks when the figure has focus;
 *                       Enter pins, Escape clears
 *   near a point        on line and dot charts, the nearest point within reach
 *                       answers the pointer, with a guide line at its x
 *   legend              hover a legend entry to highlight its series, click to
 *                       isolate it, click again to show everything
 *
 * The hoverable marks are discovered, not declared: each candidate element gets
 * a synthetic mousemove, and the ones whose listener opens the tooltip are the
 * marks. That keeps the chart renderers unchanged.
 */
import { hideTip, pinTip, tipState } from "./figure-kit";

type Target = { el: Element; cx: number; cy: number; small: boolean };

const cache = new WeakMap<SVGSVGElement, Target[]>();

function center(el: Element) {
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height };
}

function fire(el: Element, type: string, x: number, y: number) {
  el.dispatchEvent(
    new MouseEvent(type, { bubbles: type !== "mouseleave", cancelable: true, clientX: x, clientY: y, view: window })
  );
}

/** Elements whose mousemove listener opens the tooltip, in reading order. */
function targets(svg: SVGSVGElement): Target[] {
  const hit = cache.get(svg);
  if (hit) return hit;
  const found: Target[] = [];
  const seen = new Set<Element>();
  const candidates = svg.querySelectorAll("rect, circle, path, g, text, line, polygon");
  candidates.forEach(el => {
    if (el.closest("defs, pattern, clipPath, mask")) return;
    const c = center(el);
    if (c.w === 0 && c.h === 0) return;
    const before = tipState.seq;
    tipState.pinned = false;
    fire(el, "mousemove", c.x, c.y);
    if (tipState.seq !== before && tipState.target && !seen.has(tipState.target)) {
      const t = tipState.target;
      seen.add(t);
      const tc = center(t);
      found.push({ el: t, cx: tc.x, cy: tc.y, small: tc.w <= 18 && tc.h <= 18 });
    }
  });
  // Probing opened and closed the tooltip; leave it closed and unpinned.
  hideTip(true);
  found.sort((a, b) => a.cx - b.cx || a.cy - b.cy);
  cache.set(svg, found);
  return found;
}

/** Positions are measured in viewport space; refresh them on each use. */
function withCenters(ts: Target[]) {
  return ts.map(t => {
    const c = center(t.el);
    return { ...t, cx: c.x, cy: c.y };
  });
}

function show(t: Target) {
  const c = center(t.el);
  fire(t.el, "mousemove", c.x, c.y);
}

/* ------------------------------------------------------------ guide line */

function guide(node: HTMLElement): HTMLDivElement {
  let g = node.querySelector<HTMLDivElement>(":scope > .fig-guide");
  if (!g) {
    g = document.createElement("div");
    g.className = "fig-guide";
    g.setAttribute("aria-hidden", "true");
    node.appendChild(g);
  }
  return g;
}

function placeGuide(node: HTMLElement, svg: SVGSVGElement, x: number | null) {
  const g = guide(node);
  if (x === null) {
    g.style.opacity = "0";
    return;
  }
  const n = node.getBoundingClientRect();
  const s = svg.getBoundingClientRect();
  g.style.left = `${x - n.left}px`;
  g.style.top = `${s.top - n.top + 8}px`;
  g.style.height = `${s.height - 30}px`;
  g.style.opacity = "1";
}

/* --------------------------------------------------------------- legends */

function colorOf(el: Element, prop: "fill" | "stroke"): string {
  const v = getComputedStyle(el)[prop];
  return v && v !== "none" && !v.startsWith("url(") ? v : "";
}

function legendColor(item: HTMLElement): string {
  const sw = item.querySelector<HTMLElement>(".fig-swatch");
  if (!sw) return "";
  const cs = getComputedStyle(sw);
  const bg = cs.backgroundColor;
  if (bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent") return bg;
  return cs.borderTopColor;
}

function seriesMarks(svg: SVGSVGElement, color: string): Element[] {
  if (!color) return [];
  return [...svg.querySelectorAll("rect, circle, path, line, polygon")].filter(
    el => !el.closest("defs, pattern") && (colorOf(el, "fill") === color || colorOf(el, "stroke") === color)
  );
}

function wireLegend(node: HTMLElement) {
  const legend = node.querySelector<HTMLElement>(".fig-legend");
  if (!legend || legend.dataset.wired) return;
  legend.dataset.wired = "1";
  const items = [...legend.querySelectorAll<HTMLElement>(".fig-legend-item")];
  if (items.length < 2) return;
  let locked: HTMLElement | null = null;

  const apply = (active: HTMLElement | null) => {
    const svg = node.querySelector<SVGSVGElement>("svg");
    if (!svg) return;
    const all = [...svg.querySelectorAll("rect, circle, path, line, polygon")];
    all.forEach(el => el.classList.remove("fig-series-dim", "fig-series-on"));
    items.forEach(i => i.classList.toggle("is-muted", !!active && i !== active));
    if (!active) return;
    const on = new Set(seriesMarks(svg, legendColor(active)));
    if (!on.size) return;
    const others = new Set<Element>();
    items.forEach(i => {
      if (i !== active) seriesMarks(svg, legendColor(i)).forEach(m => others.add(m));
    });
    on.forEach(m => m.classList.add("fig-series-on"));
    others.forEach(m => {
      if (!on.has(m)) m.classList.add("fig-series-dim");
    });
  };

  items.forEach(item => {
    item.setAttribute("role", "button");
    item.tabIndex = 0;
    item.setAttribute("aria-pressed", "false");
    item.title = "Click to isolate this series";
    item.addEventListener("mouseenter", () => !locked && apply(item));
    item.addEventListener("mouseleave", () => !locked && apply(null));
    const toggle = () => {
      locked = locked === item ? null : item;
      items.forEach(i => i.setAttribute("aria-pressed", String(i === locked)));
      apply(locked);
    };
    item.addEventListener("click", toggle);
    item.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
  });
}

/* ------------------------------------------------------------------ wire */

export function interactive(node: HTMLElement) {
  if (node.dataset.interactive) {
    wireLegend(node);
    return;
  }
  node.dataset.interactive = "1";
  const svgNow = () => node.querySelector<SVGSVGElement>("svg");

  wireLegend(node);

  // Keyboard: the figure is one tab stop; arrows move between marks.
  const title = node.querySelector(".fig-title")?.textContent?.trim() ?? "Chart";
  node.tabIndex = 0;
  node.setAttribute("role", "group");
  node.setAttribute("aria-label", `${title}. Use the arrow keys to read each value.`);
  let cursor = -1;
  node.addEventListener("keydown", e => {
    const svg = svgNow();
    if (!svg || e.target !== node) return;
    const ts = targets(svg);
    if (!ts.length) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") cursor = (cursor + 1) % ts.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") cursor = (cursor - 1 + ts.length) % ts.length;
    else if (e.key === "Home") cursor = 0;
    else if (e.key === "End") cursor = ts.length - 1;
    else if (e.key === "Enter" && cursor >= 0) {
      e.preventDefault();
      pinTip(!tipState.pinned);
      return;
    } else if (e.key === "Escape") {
      hideTip(true);
      cursor = -1;
      return;
    } else return;
    e.preventDefault();
    tipState.pinned = false;
    show(ts[cursor]);
  });
  node.addEventListener("blur", () => {
    if (!tipState.pinned) hideTip(true);
  });

  // Click to pin (and tap to pin on touch screens).
  node.addEventListener("click", e => {
    const svg = svgNow();
    if (!svg || !(e.target instanceof Element) || !svg.contains(e.target)) return;
    const onMark = tipState.target && (tipState.target === e.target || tipState.target.contains(e.target));
    if (tipState.pinned) {
      hideTip(true);
      // Clicking a different mark moves the pin there.
      if (!onMark) {
        fire(e.target, "mousemove", e.clientX, e.clientY);
        if (tipState.target) pinTip(true);
      }
      return;
    }
    if (onMark) pinTip(true);
  });

  // Snap to the nearest point on line and dot charts.
  const REACH = 42;
  let snapped: Element | null = null;
  node.addEventListener("pointermove", e => {
    const svg = svgNow();
    if (!svg || tipState.pinned || e.pointerType === "touch") return;
    const ts = targets(svg).filter(t => t.small);
    if (ts.length < 3) return; // bars and cells are big enough to hit directly
    const direct = e.target instanceof Element && ts.some(t => t.el === e.target || t.el.contains(e.target as Node));
    let best: Target | null = null;
    let bestD = REACH;
    for (const t of withCenters(ts)) {
      const d = Math.hypot(t.cx - e.clientX, t.cy - e.clientY);
      if (d < bestD) [best, bestD] = [t, d];
    }
    if (best && !direct) {
      fire(best.el, "mousemove", e.clientX, e.clientY);
      snapped = best.el;
      placeGuide(node, svg, best.cx);
    } else if (!best && snapped) {
      fire(snapped, "mouseleave", e.clientX, e.clientY);
      snapped = null;
      placeGuide(node, svg, null);
    } else if (best) {
      placeGuide(node, svg, best.cx);
    }
  });
  node.addEventListener("pointerleave", () => {
    const svg = svgNow();
    if (snapped && !tipState.pinned) hideTip(true);
    snapped = null;
    if (svg) placeGuide(node, svg, null);
  });

  // Re-rendering charts (the GRPO slider) replace their SVG; rediscover marks.
  new MutationObserver(() => {
    const svg = svgNow();
    if (svg) cache.delete(svg);
    wireLegend(node);
  }).observe(node, { childList: true });
}
