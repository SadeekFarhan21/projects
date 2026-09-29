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

/** A one-line cue under charts that answer hover, since nothing else says so. */
export function hoverHint(node: Element) {
  if (node.querySelector(".fig-controls, .fig-hint")) return;
  const touch = window.matchMedia("(hover: none)").matches;
  const p = document.createElement("p");
  p.className = "fig-hint";
  p.textContent = touch
    ? "Tap a bar or point for exact values, tap again to let go"
    : "Hover for exact values · click to pin · arrow keys step through";
  node.appendChild(p);
}

export function caption(node: Element, text: string) {
  const c = document.createElement("figcaption");
  c.textContent = text;
  node.appendChild(c);
}

/**
 * A figure in a memo is only as good as its provenance, so the source line is a
 * separate element rather than part of the caption prose.
 */
export function sourceLine(node: Element, text: string) {
  const s = document.createElement("p");
  s.className = "fig-source";
  s.textContent = text;
  node.appendChild(s);
}

export async function json<T>(path: string): Promise<T> {
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");
  const res = await fetch(`${base}${path.replace(/^\//, "")}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}
