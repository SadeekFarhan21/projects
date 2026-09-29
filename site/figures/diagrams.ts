/**
 * Architecture diagrams, drawn from JSON specs in source/data/diagrams/NAME.json
 * and embedded as <figure data-figure="diagram:NAME"></figure>.
 *
 * They replace ASCII art in code blocks. Each form is built for one kind of
 * picture, animates in when scrolled into view, and answers the reader:
 *
 *   flow          stages on a rail: forks, merges, lanes, side outputs, loops.
 *                 Click a stage to open its details.
 *   fields        a bit or byte layout drawn to scale. Hover a field.
 *   memmap        a physical address map with permissions. Hover a region.
 *   states        a state machine. Hover a state to trace its transitions.
 *   sparse-index  an offset lookup through a sparse index. Drag the slider.
 */
import { json } from "./figure-kit";

/* ----------------------------------------------------------------- types */

type Kind = "source" | "stage" | "store" | "guard" | "output" | "note";

type Card = {
  name: string;
  role?: string;
  details?: string[];
  chain?: string[];
  cases?: { when: string; then: string }[];
  kind?: Kind;
  code?: boolean; // name is an identifier or path (default true)
};

type Step =
  | (Card & { via?: string; outputs?: (Card & { via?: string })[] })
  | { via?: string; parallel: Card[] }
  | { via?: string; lanes: { title: string; steps: Step[] }[] };

type Base = { type: string; title?: string; subtitle?: string; caption?: string };

type FlowSpec = Base & {
  type: "flow";
  steps: Step[];
  loop?: { from: number; to: number; label: string };
  extras?: { title: string; cards: Card[] };
};

type Cell = { name: string; bits?: number; range?: string; detail?: string; kind?: "empty" | "accent" | "muted"; grow?: "left" | "right" };
type Row =
  | { label?: string; cells: Cell[]; scale?: boolean }
  | { label?: string; chain: Cell[]; arrow?: string; note?: string };
type FieldsSpec = Base & { type: "fields"; rows: Row[]; unit?: string; hint?: string };

type Region = { addr: string; name: string; perm?: string; note?: string; detail?: string; kind?: "device" | "reserved" | "kernel" | "free" | "stack"; mark?: string };
type MemmapSpec = Base & { type: "memmap"; regions: Region[]; end?: { addr: string; mark?: string } };

type StateNode = { id: string; label: string; x: number; y: number; detail?: string; kind?: "start" | "end" };
type StateEdge = { from: string; to: string; label: string; bend?: number };
type StatesSpec = Base & { type: "states"; states: StateNode[]; edges: StateEdge[]; cols: number; rows: number };

type SparseSpec = Base & {
  type: "sparse-index";
  records: number[]; // record sizes in bytes
  interval: number; // bytes of log between index entries
  start?: number; // initial offset for the slider
};

/* --------------------------------------------------------------- helpers */

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function header(node: HTMLElement, spec: Base) {
  if (!spec.title && !spec.subtitle) return;
  const h = el("div", "fig-head");
  if (spec.title) h.appendChild(el("div", "fig-title", spec.title));
  if (spec.subtitle) h.appendChild(el("div", "fig-subtitle", spec.subtitle));
  node.appendChild(h);
}

function captionOf(node: HTMLElement, spec: Base) {
  if (spec.caption) node.appendChild(el("figcaption", undefined, spec.caption));
}

/** Reveal once in view; the form adds .dg-in and staggers with --i. */
function revealOnView(node: HTMLElement, onShow?: () => void) {
  if (reduced() || !("IntersectionObserver" in window)) {
    node.classList.add("dg-in", "dg-static");
    onShow?.();
    return;
  }
  const io = new IntersectionObserver(
    es =>
      es.forEach(e => {
        if (!e.isIntersecting) return;
        io.disconnect();
        node.classList.add("dg-in");
        onShow?.();
      }),
    { threshold: 0.2 }
  );
  io.observe(node);
}

function replay(node: HTMLElement, again: () => void) {
  const b = el("button", "fig-replay");
  b.type = "button";
  b.title = "Replay";
  b.setAttribute("aria-label", "Replay diagram animation");
  b.innerHTML =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 3 3 9 9 9"/></svg>';
  b.addEventListener("click", () => {
    if (reduced()) return;
    node.classList.remove("dg-in");
    void node.offsetWidth; // restart CSS transitions
    node.classList.add("dg-in");
    again();
  });
  node.appendChild(b);
}

/* ------------------------------------------------------------------ flow */

function card(c: Card, i: number): HTMLElement {
  const box = el("div", `dg-card dg-kind-${c.kind ?? "stage"}`);
  box.style.setProperty("--i", String(i));
  const name = el(c.code === false ? "span" : "code", "dg-name", c.name);
  box.appendChild(name);
  if (c.role) box.appendChild(el("div", "dg-role", c.role));
  if (c.chain?.length) {
    const ch = el("div", "dg-chain");
    c.chain.forEach((link, k) => {
      if (k) ch.appendChild(el("span", "dg-chain-arrow", "→"));
      ch.appendChild(el("code", "dg-chip", link));
    });
    box.appendChild(ch);
  }
  const more = (c.details?.length ?? 0) + (c.cases?.length ?? 0);
  if (more) {
    const body = el("div", "dg-more");
    if (c.details?.length) {
      const ul = el("ul", "dg-details");
      c.details.forEach(d => ul.appendChild(el("li", undefined, d)));
      body.appendChild(ul);
    }
    if (c.cases?.length) {
      const dl = el("dl", "dg-cases");
      c.cases.forEach(k => {
        dl.appendChild(el("dt", undefined, k.when));
        dl.appendChild(el("dd", undefined, k.then));
      });
      body.appendChild(dl);
    }
    box.appendChild(body);
    box.classList.add("has-more");
    box.tabIndex = 0;
    box.setAttribute("role", "button");
    box.setAttribute("aria-expanded", "false");
    const toggle = () => {
      const open = box.classList.toggle("is-open");
      box.setAttribute("aria-expanded", String(open));
    };
    box.addEventListener("click", e => {
      if ((e.target as Element).closest("a")) return;
      toggle();
    });
    box.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
    box.appendChild(el("span", "dg-toggle", ""));
  }
  return box;
}

let order = 0;

function stepEl(s: Step): HTMLElement {
  const li = el("li", "dg-step");
  li.style.setProperty("--i", String(order++));
  if (s.via) li.appendChild(el("div", "dg-via", s.via));
  else li.classList.add("no-via");

  if ("parallel" in s) {
    li.classList.add("dg-parallel");
    const row = el("div", "dg-row");
    row.style.setProperty("--n", String(s.parallel.length));
    s.parallel.forEach(c => {
      const col = el("div", "dg-col");
      col.appendChild(card(c, order++));
      row.appendChild(col);
    });
    li.appendChild(row);
    return li;
  }
  if ("lanes" in s) {
    li.classList.add("dg-lanes");
    const row = el("div", "dg-row");
    row.style.setProperty("--n", String(s.lanes.length));
    s.lanes.forEach(lane => {
      const col = el("div", "dg-lane");
      col.appendChild(el("div", "dg-lane-title", lane.title));
      const ol = el("ol", "dg-steps");
      lane.steps.forEach(st => ol.appendChild(stepEl(st)));
      col.appendChild(ol);
      row.appendChild(col);
    });
    li.appendChild(row);
    return li;
  }
  const wrap = el("div", "dg-main");
  wrap.appendChild(card(s, order));
  if (s.outputs?.length) {
    const outs = el("div", "dg-outputs");
    s.outputs.forEach(o => {
      const line = el("div", "dg-out");
      if (o.via) line.appendChild(el("div", "dg-out-via", o.via));
      line.appendChild(card({ kind: "output", ...o }, order));
      outs.appendChild(line);
    });
    wrap.appendChild(outs);
    wrap.classList.add("has-outputs");
  }
  li.appendChild(wrap);
  return li;
}

function flow(node: HTMLElement, d: FlowSpec) {
  order = 0;
  node.classList.add("dg", "dg-flow");
  header(node, d);

  const tools = el("div", "dg-tools");
  const expand = el("button", "dg-expand", "Expand all");
  expand.type = "button";
  tools.appendChild(expand);
  node.appendChild(tools);

  const ol = el("ol", "dg-steps dg-root");
  d.steps.forEach(s => ol.appendChild(stepEl(s)));
  const stage = el("div", "dg-stage");
  stage.appendChild(ol);
  node.appendChild(stage);

  if (d.extras) {
    const ex = el("div", "dg-extras");
    ex.appendChild(el("div", "dg-extras-title", d.extras.title));
    const grid = el("div", "dg-extras-grid");
    d.extras.cards.forEach(c => grid.appendChild(card({ kind: "guard", ...c }, order++)));
    ex.appendChild(grid);
    node.appendChild(ex);
  }
  captionOf(node, d);

  const cards = () => [...node.querySelectorAll<HTMLElement>(".dg-card.has-more")];
  if (!cards().length) tools.remove();
  expand.addEventListener("click", () => {
    const openAll = cards().some(c => !c.classList.contains("is-open"));
    cards().forEach(c => {
      c.classList.toggle("is-open", openAll);
      c.setAttribute("aria-expanded", String(openAll));
    });
    expand.textContent = openAll ? "Collapse all" : "Expand all";
    if (d.loop) requestAnimationFrame(() => drawLoop(node, stage, d.loop!));
  });

  if (d.loop) {
    const draw = () => drawLoop(node, stage, d.loop!);
    requestAnimationFrame(draw);
    window.addEventListener("resize", draw);
    node.addEventListener("transitionend", e => {
      if ((e.target as Element).classList?.contains("dg-card")) draw();
    });
  }
  node.style.setProperty("--steps", String(order));
  revealOnView(node);
  replay(node, () => {});
}

/** A return arrow on the left edge for event loops: from one step back to another. */
function drawLoop(node: HTMLElement, stage: HTMLElement, loop: { from: number; to: number; label: string }) {
  const steps = [...stage.querySelectorAll<HTMLElement>(":scope > .dg-root > .dg-step")];
  const a = steps[loop.to]?.querySelector<HTMLElement>(".dg-card");
  const b = steps[loop.from]?.querySelector<HTMLElement>(".dg-card");
  if (!a || !b) return;
  let arc = stage.querySelector<HTMLElement>(".dg-loop");
  if (!arc) {
    arc = el("div", "dg-loop");
    arc.appendChild(el("span", "dg-loop-label", loop.label));
    stage.appendChild(arc);
  }
  const s = stage.getBoundingClientRect();
  const ra = a.getBoundingClientRect();
  const rb = b.getBoundingClientRect();
  const left = Math.min(ra.left, rb.left) - s.left;
  arc.style.top = `${ra.top - s.top + ra.height / 2}px`;
  arc.style.height = `${rb.top + rb.height / 2 - (ra.top + ra.height / 2)}px`;
  arc.style.left = `${Math.max(0, left - 22)}px`;
}

/* ---------------------------------------------------------------- fields */

function detailPane(node: HTMLElement, hint: string) {
  const p = el("div", "dg-detail");
  p.setAttribute("aria-live", "polite");
  p.textContent = hint;
  node.appendChild(p);
  return (text: string | null) => {
    p.textContent = text ?? hint;
    p.classList.toggle("is-empty", !text);
  };
}

function fields(node: HTMLElement, d: FieldsSpec) {
  node.classList.add("dg", "dg-fields");
  header(node, d);
  const rowsEl = el("div", "dg-rows");
  node.appendChild(rowsEl);
  const say = detailPane(node, d.hint ?? "Hover or tap a field to see what it holds");
  let i = 0;
  const bind = (c: HTMLElement, text: string | undefined, name: string) => {
    c.tabIndex = 0;
    const msg = text ? `${name}. ${text}` : name;
    c.addEventListener("mouseenter", () => say(msg));
    c.addEventListener("focus", () => say(msg));
    c.addEventListener("click", () => say(msg));
    c.addEventListener("mouseleave", () => say(null));
    c.addEventListener("blur", () => say(null));
  };

  d.rows.forEach(r => {
    const row = el("div", "dg-frow");
    if (r.label) row.appendChild(el("div", "dg-frow-label", r.label));
    const track = el("div", "chain" in r ? "dg-ftrack dg-fchain" : "dg-ftrack");
    if ("chain" in r) {
      r.chain.forEach((c, k) => {
        if (k) track.appendChild(el("span", "dg-chain-arrow", r.arrow ?? "→"));
        const chip = el("div", `dg-fchip ${c.kind ? "is-" + c.kind : ""}`);
        chip.style.setProperty("--i", String(i++));
        chip.appendChild(el("code", undefined, c.name));
        bind(chip, c.detail, c.name);
        track.appendChild(chip);
      });
      row.appendChild(track);
      if (r.note) row.appendChild(el("div", "dg-frow-note", r.note));
    } else {
      r.cells.forEach(c => {
        const cell = el("div", `dg-fcell ${c.kind ? "is-" + c.kind : ""}`);
        cell.style.setProperty("--i", String(i++));
        cell.style.flexGrow = String(r.scale === false ? 1 : c.bits ?? 1);
        if (c.range) cell.appendChild(el("span", "dg-frange", c.range));
        const nm = el("code", "dg-fname", c.name);
        if (c.grow === "right") nm.textContent = `${c.name}  →`;
        if (c.grow === "left") nm.textContent = `←  ${c.name}`;
        cell.appendChild(nm);
        if (c.bits && r.scale !== false) cell.appendChild(el("span", "dg-fbits", `${c.bits} ${d.unit ?? "bits"}`));
        bind(cell, c.detail, c.name);
        cell.title = c.detail ?? c.name;
        track.appendChild(cell);
      });
      row.appendChild(track);
    }
    rowsEl.appendChild(row);
  });
  captionOf(node, d);
  revealOnView(node);
  replay(node, () => {});
}

/* ---------------------------------------------------------------- memmap */

function memmap(node: HTMLElement, d: MemmapSpec) {
  node.classList.add("dg", "dg-memmap");
  header(node, d);
  const list = el("div", "dg-regions");
  node.appendChild(list);
  const say = detailPane(node, "Hover or tap a region for its purpose");
  d.regions.forEach((r, i) => {
    const row = el("div", `dg-region is-${r.kind ?? "kernel"}`);
    row.style.setProperty("--i", String(i));
    row.appendChild(el("code", "dg-addr", r.addr));
    const block = el("div", "dg-block");
    block.appendChild(el("span", "dg-rname", r.name));
    if (r.perm) block.appendChild(el("code", "dg-perm", r.perm));
    row.appendChild(block);
    row.appendChild(el("span", "dg-rnote", r.note ?? ""));
    if (r.mark) row.appendChild(el("code", "dg-mark", r.mark));
    row.tabIndex = 0;
    const msg = `${r.name} at ${r.addr}${r.detail ? `. ${r.detail}` : r.note ? `. ${r.note}` : ""}`;
    row.addEventListener("mouseenter", () => say(msg));
    row.addEventListener("focus", () => say(msg));
    row.addEventListener("click", () => say(msg));
    row.addEventListener("mouseleave", () => say(null));
    list.appendChild(row);
  });
  if (d.end) {
    const row = el("div", "dg-region dg-end");
    row.appendChild(el("code", "dg-addr", d.end.addr));
    row.appendChild(el("div", "dg-block dg-block-end"));
    row.appendChild(el("span", "dg-rnote", ""));
    if (d.end.mark) row.appendChild(el("code", "dg-mark", d.end.mark));
    list.appendChild(row);
  }
  captionOf(node, d);
  revealOnView(node);
  replay(node, () => {});
}

/* ---------------------------------------------------------------- states */

const SVGNS = "http://www.w3.org/2000/svg";
function s<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function states(node: HTMLElement, d: StatesSpec) {
  node.classList.add("dg", "dg-states");
  header(node, d);
  const CW = 220,
    RH = 96,
    NW = 128,
    NH = 38,
    PAD = 34;
  const W = d.cols * CW + PAD * 2,
    H = d.rows * RH + PAD * 2 - 24;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": d.title ?? "State machine" });
  const defs = s("defs");
  const marker = s("marker", { id: `dg-arrow-${node.dataset.figure}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  marker.appendChild(s("path", { d: "M0,0 L10,5 L0,10 z", class: "dg-arrowhead" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const pos = new Map(d.states.map(n => [n.id, { x: PAD + n.x * CW + CW / 2, y: PAD + n.y * RH + NH / 2 }]));
  const edgesG = s("g", { class: "dg-edges" });
  const nodesG = s("g", { class: "dg-nodes" });
  svg.append(edgesG, nodesG);

  const edgeEls: { e: StateEdge; g: SVGGElement }[] = [];
  d.edges.forEach((e, i) => {
    const a = pos.get(e.from)!,
      b = pos.get(e.to)!;
    // Leave from and arrive at the node boundary along the line between centres.
    const dx = b.x - a.x,
      dy = b.y - a.y;
    const clip = (dx0: number, dy0: number) => {
      const tx = dx0 === 0 ? Infinity : NW / 2 / Math.abs(dx0);
      const ty = dy0 === 0 ? Infinity : NH / 2 / Math.abs(dy0);
      return Math.min(tx, ty);
    };
    const t = clip(dx, dy);
    const sx = a.x + dx * t,
      sy = a.y + dy * t,
      ex = b.x - dx * t,
      ey = b.y - dy * t;
    const bend = e.bend ?? 0;
    const mx = (sx + ex) / 2 - dy * bend * 0.35,
      my = (sy + ey) / 2 + dx * bend * 0.35;
    const g = s("g", { class: "dg-edge" }) as SVGGElement;
    g.style.setProperty("--i", String(i));
    const path = s("path", { d: `M${sx},${sy} Q${mx},${my} ${ex},${ey}`, "marker-end": `url(#dg-arrow-${node.dataset.figure})`, class: "dg-edge-line" });
    const lx = (sx + 2 * mx + ex) / 4,
      ly = (sy + 2 * my + ey) / 4;
    const label = s("text", { x: lx, y: ly - 6, class: "dg-edge-label", "text-anchor": "middle" });
    label.textContent = e.label;
    g.append(path, label);
    edgesG.appendChild(g);
    edgeEls.push({ e, g });
  });

  const say = detailPane(node, "Hover or tap a state to trace its transitions");
  d.states.forEach((n, i) => {
    const p = pos.get(n.id)!;
    const g = s("g", { class: `dg-node ${n.kind ? "is-" + n.kind : ""}`, tabindex: 0, transform: `translate(${p.x - NW / 2},${p.y - NH / 2})` }) as SVGGElement;
    g.style.setProperty("--i", String(i));
    g.append(s("rect", { width: NW, height: NH, rx: NH / 2 }));
    const tx = s("text", { x: NW / 2, y: NH / 2 + 4.5, "text-anchor": "middle" });
    tx.textContent = n.label;
    g.appendChild(tx);
    const focus = (on: boolean) => {
      svg.classList.toggle("is-tracing", on);
      g.classList.toggle("is-hot", on);
      const ins: string[] = [],
        outs: string[] = [];
      edgeEls.forEach(({ e, g: eg }) => {
        const hit = on && (e.from === n.id || e.to === n.id);
        eg.classList.toggle("is-hot", hit);
        if (hit && e.from === n.id) outs.push(`${e.label} → ${e.to}`);
        if (hit && e.to === n.id) ins.push(`${e.from} → ${e.label}`);
      });
      say(on ? `${n.label}${n.detail ? `. ${n.detail}` : ""}${ins.length ? `. In: ${ins.join("; ")}` : ""}${outs.length ? `. Out: ${outs.join("; ")}` : ""}` : null);
    };
    g.addEventListener("mouseenter", () => focus(true));
    g.addEventListener("mouseleave", () => focus(false));
    g.addEventListener("focus", () => focus(true));
    g.addEventListener("blur", () => focus(false));
    g.addEventListener("click", () => focus(true));
    nodesG.appendChild(g);
  });

  const wrap = el("div", "dg-svg");
  wrap.appendChild(svg);
  node.insertBefore(wrap, node.querySelector(".dg-detail"));
  captionOf(node, d);
  const drawEdges = () => {
    if (reduced()) return;
    edgeEls.forEach(({ g }, i) => {
      const path = g.querySelector("path")!;
      const len = path.getTotalLength();
      path.animate([{ strokeDasharray: `${len}`, strokeDashoffset: len }, { strokeDasharray: `${len}`, strokeDashoffset: 0 }], {
        duration: 600,
        delay: 300 + i * 120,
        easing: "cubic-bezier(0.45, 0, 0.2, 1)",
        fill: "backwards",
      });
    });
  };
  revealOnView(node, drawEdges);
  replay(node, drawEdges);
}

/* ---------------------------------------------------------- sparse index */

function sparseIndex(node: HTMLElement, d: SparseSpec) {
  node.classList.add("dg", "dg-sparse");
  header(node, d);

  // Byte position of each record, and the index entries: one whenever at
  // least `interval` bytes of log have passed since the last entry.
  const pos: number[] = [];
  let at = 0;
  d.records.forEach(sz => {
    pos.push(at);
    at += sz;
  });
  const total = at;
  const entries: { rel: number; pos: number }[] = [];
  let last = -Infinity;
  pos.forEach((p, i) => {
    if (p - last >= d.interval || i === 0) {
      entries.push({ rel: i, pos: p });
      last = p;
    }
  });

  const layout = el("div", "dg-sp");
  const idx = el("div", "dg-sp-index");
  idx.appendChild(el("div", "dg-sp-cap", "index file"));
  const table = el("table", "dg-sp-table");
  const thead = el("thead");
  const hr = el("tr");
  hr.append(el("th", undefined, "rel_offset"), el("th", undefined, "position"));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el("tbody");
  const rowEls = entries.map(e => {
    const tr = el("tr");
    tr.append(el("td", undefined, String(e.rel)), el("td", undefined, e.pos.toLocaleString("en-US")));
    tbody.appendChild(tr);
    return tr;
  });
  table.appendChild(tbody);
  idx.appendChild(table);

  const log = el("div", "dg-sp-log");
  log.appendChild(el("div", "dg-sp-cap", `log file, ${total.toLocaleString("en-US")} bytes`));
  const bar = el("div", "dg-sp-bar");
  const recEls = d.records.map((sz, i) => {
    const r = el("div", "dg-sp-rec");
    r.style.flexGrow = String(sz);
    r.style.setProperty("--i", String(i));
    r.title = `record ${i}, ${sz} bytes at ${pos[i].toLocaleString("en-US")}`;
    r.appendChild(el("span", undefined, String(i)));
    if (entries.some(e => e.rel === i)) r.classList.add("is-indexed");
    bar.appendChild(r);
    return r;
  });
  log.appendChild(bar);
  layout.append(idx, log);
  node.appendChild(layout);

  const controls = el("div", "fig-controls dg-sp-controls");
  const lab = el("label", undefined, "fetch offset");
  const input = el("input") as HTMLInputElement;
  input.type = "range";
  input.min = "0";
  input.max = String(d.records.length - 1);
  input.value = String(d.start ?? Math.floor(d.records.length * 0.6));
  input.setAttribute("aria-label", "Offset to fetch");
  const out = el("span", "fig-value");
  lab.appendChild(input);
  controls.append(lab, out);
  node.insertBefore(controls, layout);

  const steps = el("ol", "dg-sp-steps");
  node.appendChild(steps);

  const update = () => {
    const n = Number(input.value);
    let k = 0;
    entries.forEach((e, j) => {
      if (e.rel <= n) k = j;
    });
    const floor = entries[k];
    out.textContent = `n = ${n}`;
    rowEls.forEach((r, j) => r.classList.toggle("is-hit", j === k));
    recEls.forEach((r, i) => {
      r.classList.toggle("is-skip", i >= floor.rel && i < n);
      r.classList.toggle("is-target", i === n);
      r.classList.toggle("is-after", i > n);
    });
    const skipped = n - floor.rel;
    steps.innerHTML = "";
    [
      `Binary search the index for the last entry at or below ${n}: entry (${floor.rel}, ${floor.pos.toLocaleString("en-US")}).`,
      `pread the log from byte ${floor.pos.toLocaleString("en-US")}.`,
      skipped
        ? `Skip ${skipped} record header${skipped > 1 ? "s" : ""} (${floor.rel} to ${n - 1}), reading only lengths.`
        : `No headers to skip, the entry points straight at record ${n}.`,
      `Return records from ${n} onward.`,
    ].forEach(t => steps.appendChild(el("li", undefined, t)));
  };
  input.addEventListener("input", update);
  update();
  captionOf(node, d);
  revealOnView(node);
  replay(node, () => {});
}

/* -------------------------------------------------------------- dispatch */

export async function diagramFigure(node: HTMLElement, name: string) {
  const spec = await json<Base>(`data/diagrams/${name}.json`);
  node.dataset.motion = "self";
  switch (spec.type) {
    case "flow":
      return flow(node, spec as FlowSpec);
    case "fields":
      return fields(node, spec as FieldsSpec);
    case "memmap":
      return memmap(node, spec as MemmapSpec);
    case "states":
      return states(node, spec as StatesSpec);
    case "sparse-index":
      return sparseIndex(node, spec as SparseSpec);
    default:
      throw new Error(`unknown diagram type ${spec.type}`);
  }
}
