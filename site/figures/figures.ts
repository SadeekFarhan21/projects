/**
 * D3 figures for the technical posts. Shared plumbing lives in figure-kit.ts;
 * the venture memos' charts live in venture-figures.ts and dispatch from here.
 *
 * A post drops a placeholder into its markdown:
 *
 *   <figure data-figure="induction-heatmap"></figure>
 *
 * and this module fills it in. The numbers come from /data/*.json, written by the
 * generator script in each project repo (projects/<name>/figures/make_figures.py),
 * so a figure can never drift from the run that produced it.
 */
import { select } from "d3-selection";
import { scaleLinear, scaleSequential } from "d3-scale";
import { interpolateRgb } from "d3-interpolate";
import { max } from "d3-array";
import {
  theme,
  fmt,
  showTip,
  hideTip,
  caption,
  sourceLine,
  heading,
  legend,
  responsive,
  textWidth,
  json,
  hoverHint,
} from "./figure-kit";
import { animateIn } from "./motion";
import { interactive } from "./interact";
import { ventureFigure, specFigure } from "./venture-figures";

/* ---------------------------------------------------------------- heatmap */

type Cell = { layer: number; head: number; score: number };

type Scores = {
  model: string;
  source: string;
  n_layers: number;
  n_heads: number;
  gap: number;
  measurement: string;
  canonical: { layer: number; head: number }[];
  scores: Cell[];
};

type Svg = ReturnType<typeof select<SVGSVGElement, unknown>>;

/** Size a responsive chart's <svg> (figure-kit responsive()) in CSS px. */
function sized(el: SVGSVGElement, w: number, h: number, label: string): Svg {
  return select(el)
    .attr("viewBox", `0 0 ${w} ${h}`)
    .attr("role", "img")
    .attr("aria-label", label) as unknown as Svg;
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

async function inductionHeatmap(node: Element) {
  const d = await json<Scores>("data/induction-scores.json");
  const t = theme();
  heading(node, {
    title: `Induction Score for Every Head in ${d.model.replace(/\b\w/g, c => c.toUpperCase())}`,
    subtitle: `${cap(d.measurement)}, by layer and head`,
  });

  const hi = max(d.scores, (s: Cell) => s.score) ?? 1;
  const color = scaleSequential<string>(
    interpolateRgb("#eef3f7", t.accent)
  ).domain([0, hi]);
  const isCanonical = new Set(d.canonical.map(c => `${c.layer}-${c.head}`));
  const aria =
    `Induction score for all ${d.scores.length} attention heads of GPT-2 small, by layer and head. ` +
    `The five canonical induction heads score above 0.80; the next head scores 0.517.`;

  responsive(node, (el, L) => {
    const ap = L.axisPx;
    const gapPx = L.compact ? 2 : 3,
      padL = Math.ceil(textWidth("Layer", ap)) + 10,
      // Axis titles: "Head" over the column numbers, "Layer" at the corner.
      padT = 2 * ap + 14,
      legendW = Math.ceil(textWidth("0.00", ap)) + 32;
    // Square cells as large as the width allows, up to 30px.
    const cell = Math.max(
      12,
      Math.min(30, Math.floor((L.w - padL - legendW) / d.n_heads) - gapPx)
    );
    const step = cell + gapPx;
    const w = padL + d.n_heads * step + legendW;
    const h = padT + d.n_layers * step + 4;
    const svg = sized(el, w, h, aria);
    // Drawn at its own width, left-aligned under the title, not stretched.
    el.style.width = `${w}px`;
    el.style.maxWidth = "100%";

    svg
      .append("text")
      .attr("class", "fig-axis fig-axis-title")
      .attr("x", padL)
      .attr("y", ap)
      .text("Head");
    svg
      .append("text")
      .attr("class", "fig-axis fig-axis-title")
      .attr("x", padL - 8)
      .attr("y", padT - 8)
      .attr("text-anchor", "end")
      .text("Layer");
    for (let head = 0; head < d.n_heads; head++) {
      svg
        .append("text")
        .attr("x", padL + head * step + cell / 2)
        .attr("y", padT - 8)
        .attr("text-anchor", "middle")
        .attr("class", "fig-axis")
        .text(head);
    }
    for (let layer = 0; layer < d.n_layers; layer++) {
      svg
        .append("text")
        .attr("x", padL - 8)
        .attr("y", padT + layer * step + cell / 2 + ap * 0.35)
        .attr("text-anchor", "end")
        .attr("class", "fig-axis")
        .text(`L${layer}`);
    }

    svg
      .selectAll<SVGRectElement, Cell>("rect.cell")
      .data(d.scores)
      .join("rect")
      .attr("class", "cell")
      .attr("x", (s: Cell) => padL + s.head * step)
      .attr("y", (s: Cell) => padT + s.layer * step)
      .attr("width", cell)
      .attr("height", cell)
      .attr("rx", 3)
      .attr("fill", (s: Cell) => color(s.score))
      .attr("stroke", (s: Cell) =>
        isCanonical.has(`${s.layer}-${s.head}`) ? t.fg : "transparent"
      )
      .attr("stroke-width", 1.75)
      .on("mousemove", (event: MouseEvent, s: Cell) =>
        showTip(
          `<strong>L${s.layer}H${s.head}</strong><br>Induction score ${s.score.toFixed(3)}` +
            (isCanonical.has(`${s.layer}-${s.head}`)
              ? "<br><em>Canonical induction head</em>"
              : ""),
          event
        )
      )
      .on("mouseleave", hideTip);

    // sequential legend
    const lx = padL + d.n_heads * step + 12;
    const lh = Math.min(132, d.n_layers * step - gapPx);
    const rampId = `fig-ramp-${Math.random().toString(36).slice(2)}`;
    const grad = svg
      .append("defs")
      .append("linearGradient")
      .attr("id", rampId)
      .attr("x1", "0")
      .attr("y1", "1")
      .attr("x2", "0")
      .attr("y2", "0");
    for (let i = 0; i <= 10; i++)
      grad
        .append("stop")
        .attr("offset", `${i * 10}%`)
        .attr("stop-color", color((hi * i) / 10));
    svg
      .append("rect")
      .attr("x", lx)
      .attr("y", padT)
      .attr("width", 11)
      .attr("height", lh)
      .attr("rx", 2)
      .attr("fill", `url(#${rampId})`)
      .attr("stroke", t.border);
    (
      [
        [0, hi],
        [0.5, hi / 2],
        [1, 0],
      ] as [number, number][]
    ).forEach(([pos, val]) =>
      svg
        .append("text")
        .attr("x", lx + 17)
        .attr("y", padT + pos * lh + ap * 0.35)
        .attr("class", "fig-axis")
        .text(val.toFixed(2))
    );
  });

  caption(
    node,
    `Induction score for every head in GPT-2 small, ${d.measurement}. Outlined cells are the five heads the literature names.`
  );
  sourceLine(node, d.source);
}

/* ----------------------------------------------------------- ranked heads */

async function inductionRanked(node: Element) {
  const d = await json<Scores>("data/induction-scores.json");
  const t = theme();
  const rows = [...d.scores].sort((a, b) => b.score - a.score).slice(0, 14);
  heading(node, {
    title: "The Fourteen Highest-Scoring Heads",
    subtitle: `Induction score by head, highest first. The dashed line marks the gap after rank 5`,
  });
  const aria = `The fourteen highest-scoring heads. The top five score ${rows[4].score.toFixed(3)} to ${rows[0].score.toFixed(3)}; the sixth scores ${rows[5].score.toFixed(3)}, a gap of ${d.gap.toFixed(3)}.`;
  const name = (s: Cell) => `L${s.layer}H${s.head}`;

  responsive(node, (el, L) => {
    const lp = L.labelPx;
    const rowH = Math.max(24, lp + 11),
      padT = 6,
      padL = Math.ceil(Math.max(...rows.map(s => textWidth(name(s), lp, 450)))) + 12,
      padR = Math.ceil(textWidth("0.000", lp)) + 14;
    const w = L.w;
    const h = padT + rows.length * rowH + 4;
    const x = scaleLinear()
      .domain([0, rows[0].score])
      .range([0, w - padL - padR]);
    const svg = sized(el, w, h, aria);
    const barH = 13,
      barY = (rowH - barH) / 2,
      textY = rowH / 2 + lp * 0.35;

    const g = svg
      .selectAll<SVGGElement, Cell>("g.row")
      .data(rows)
      .join("g")
      .attr("class", "row")
      .attr(
        "transform",
        (_d: Cell, i: number) => `translate(0,${padT + i * rowH})`
      )
      .on("mousemove", (event: MouseEvent, s: Cell) =>
        showTip(
          `<strong>${name(s)}</strong><br>Induction score ${s.score.toFixed(3)}`,
          event
        )
      )
      .on("mouseleave", hideTip);

    g.append("rect") // hit target, wider than the mark
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", w)
      .attr("height", rowH)
      .attr("fill", "transparent");
    g.append("text")
      .attr("x", padL - 10)
      .attr("y", textY)
      .attr("text-anchor", "end")
      .attr("class", "fig-label")
      .attr("font-weight", (_d: Cell, i: number) => (i < 5 ? 650 : 450))
      .text(name);
    g.append("rect")
      .attr("x", padL)
      .attr("y", barY)
      .attr("width", (s: Cell) => x(s.score))
      .attr("height", barH)
      .attr("rx", 4)
      .attr("fill", (_d: Cell, i: number) => (i < 5 ? t.accent : "#c9d7e2"));
    g.append("text")
      .attr("x", (s: Cell) => padL + x(s.score) + 8)
      .attr("y", textY)
      .attr("class", "fig-label fig-num")
      .text((s: Cell) => s.score.toFixed(3));

    const gy = padT + 5 * rowH;
    svg
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", gy)
      .attr("y2", gy)
      .attr("stroke", t.fg)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3 3");
    svg
      .append("text")
      .attr("x", w)
      // Below the line, where the lower bars leave the right side empty, so the
      // label cannot collide with the rank-5 value however long that bar gets.
      .attr("y", gy + lp + 4)
      .attr("text-anchor", "end")
      .attr("class", "fig-label")
      .text(`Gap of ${d.gap.toFixed(3)}`);
  });

  legend(node, [
    { label: "Canonical induction heads", swatch: t.accent },
    { label: "Next highest", swatch: "#c9d7e2" },
  ]);
  caption(
    node,
    `The top five heads and the drop below them. The ${d.gap.toFixed(3)} gap after rank 5 is what makes "the induction heads" a set rather than a cutoff someone chose.`
  );
  sourceLine(node, d.source);
}

/* -------------------------------------------------------- parameter budget */

type Group = {
  label: string;
  params: number;
  share: number;
  vocab_bound: boolean;
};

type Params = {
  source: string;
  total_params: number;
  final_loss: number;
  config: Record<string, number>;
  groups: Group[];
};

async function gptParameters(node: Element) {
  const d = await json<Params>("data/gpt-parameters.json");
  const t = theme();
  const VOCAB = "#8fb8d0";
  heading(node, {
    title: "Where the Parameters Sit",
    subtitle: `${fmt.format(d.total_params)} parameters by component, share of the model after each count`,
  });
  const aria =
    `Parameter budget of the ${fmt.format(d.total_params)} parameter model by component. ` +
    d.groups
      .map((g: Group) => `${g.label} ${fmt.format(g.params)}, ${g.share}%`)
      .join("; ");
  const value = (r: Group) => `${fmt.format(r.params)} · ${r.share.toFixed(1)}%`;

  responsive(node, (el, L) => {
    const lp = L.labelPx;
    const labelW = Math.max(...d.groups.map(g => textWidth(g.label, lp, 450)));
    const padR = Math.ceil(Math.max(...d.groups.map(g => textWidth(value(g), lp)))) + 14;
    const inlineL = Math.ceil(labelW) + 14;
    // Same rule as the spec bar charts: labels above the bars when beside
    // them would leave the bars under 45% of the width.
    const stacked = L.w - inlineL - padR < L.w * 0.45;
    const padL = stacked ? 0 : inlineL;
    const rowH = stacked ? lp + 25 : Math.max(30, lp + 17),
      barH = 15,
      barY = stacked ? lp + 5 : (rowH - barH) / 2,
      textY = barY + barH / 2 + lp * 0.35,
      padT = 4;
    const w = L.w;
    const h = padT + d.groups.length * rowH + 4;
    const x = scaleLinear()
      .domain([0, max(d.groups, (g: Group) => g.params) ?? 1])
      .range([0, w - padL - padR]);
    const svg = sized(el, w, h, aria);

    const g = svg
      .selectAll<SVGGElement, Group>("g.row")
      .data(d.groups)
      .join("g")
      .attr(
        "transform",
        (_d: Group, i: number) => `translate(0,${padT + i * rowH})`
      )
      .on("mousemove", (event: MouseEvent, r: Group) =>
        showTip(
          `<strong>${r.label}</strong><br>${fmt.format(r.params)} parameters<br>${r.share}% of the model`,
          event
        )
      )
      .on("mouseleave", hideTip);

    g.append("rect")
      .attr("width", w)
      .attr("height", rowH)
      .attr("fill", "transparent");
    g.append("text")
      .attr("x", stacked ? 0 : padL - 12)
      .attr("y", stacked ? lp : textY)
      .attr("text-anchor", stacked ? "start" : "end")
      .attr("class", "fig-label")
      .text((r: Group) => r.label);
    g.append("rect")
      .attr("x", padL)
      .attr("y", barY)
      .attr("width", (r: Group) => Math.max(2, x(r.params)))
      .attr("height", barH)
      .attr("rx", 4)
      .attr("fill", (r: Group) => (r.vocab_bound ? VOCAB : t.accent));
    g.append("text")
      .attr("x", (r: Group) => padL + Math.max(2, x(r.params)) + 8)
      .attr("y", textY)
      .attr("class", "fig-label fig-num")
      .text(value);
  });

  legend(node, [
    { label: "Transformer blocks and norms", swatch: t.accent },
    { label: "Vocabulary-bound tensors", swatch: VOCAB },
  ]);
  caption(
    node,
    `Where the ${fmt.format(d.total_params)} parameters sit, read from the shipped checkpoint (${d.config.n_layers} blocks, ${d.config.num_heads} heads, d_model ${d.config.n_embed}, vocab ${d.config.vocab_size}). Lighter bars are the two vocabulary-bound tensors; the MLPs hold more than the attention they surround.`
  );
  sourceLine(node, d.source);
}

/* ------------------------------------------------- GRPO advantage explorer */

/**
 * Not a measurement: this is the GRPO advantage definition evaluated for a group
 * of 8 rollouts, k of them correct. It shows the post's claim directly: the
 * signal is the within-group spread, so it vanishes at k=0 and k=8 alike.
 */
function grpoAdvantage(node: Element) {
  const t = theme();
  const G = 8;
  const FAIL = "#b9c3cc";
  // The slider is this chart's keyboard control; it has no marks to step through.
  (node as HTMLElement).dataset.keys = "off";
  heading(node, {
    title: "GRPO Advantages for One Group of Eight Rollouts",
    subtitle: "The advantage definition evaluated for k correct rollouts, not a measurement",
  });
  // The control row sits above the chart it drives.
  const controls = document.createElement("div");
  controls.className = "fig-controls";
  node.appendChild(controls);

  const input = document.createElement("input");
  input.type = "range";
  input.min = "0";
  input.max = String(G);
  input.step = "1";
  input.value = "3";
  input.setAttribute("aria-label", "Rollouts solved correctly, out of 8");
  input.style.minHeight = "24px"; // WCAG 2.5.8 minimum target size
  const readout = document.createElement("span");
  readout.className = "fig-value";
  controls.append(
    Object.assign(document.createElement("label"), {
      textContent: "Correct rollouts",
    }),
    input,
    readout
  );

  // Set by each (re)draw of the SVG; update() moves the marks for a new k.
  let update: (k: number) => void = () => {};

  responsive(node, (el, L) => {
    const lp = L.labelPx,
      ap = L.axisPx;
    const w = L.w;
    // Advantages reach +-2.65 (one solved or one failed of eight); SCALE px
    // per unit keeps the tallest bar clear of the reward row.
    const SCALE = 30,
      reach = Math.ceil(2.7 * SCALE);
    const noteY = lp;
    const yReward = noteY + 30;
    const yAdv = yReward + 16 + reach;
    const h = yAdv + reach + 4;
    const svg = sized(
      el,
      w,
      h,
      "GRPO advantages for a group of eight rollouts as the number of correct rollouts varies. " +
        "At zero correct and at eight correct the rewards are identical, the standard deviation is zero, and every advantage is zero."
    );
    const padL = Math.ceil(textWidth("Advantage", lp)) + 14;
    const x = scaleLinear()
      .domain([0, G - 1])
      .range([padL + 12, w - 12]);
    const half = Math.min(9, ((w - padL - 24) / (G - 1)) * 0.3);

    svg
      .append("text")
      .attr("x", padL - 14)
      .attr("y", yReward + lp * 0.35)
      .attr("text-anchor", "end")
      .attr("class", "fig-label")
      .text("Reward");
    svg
      .append("text")
      .attr("x", padL - 14)
      .attr("y", yAdv + lp * 0.35)
      .attr("text-anchor", "end")
      .attr("class", "fig-label")
      .text("Advantage");
    svg
      .append("line")
      .attr("x1", padL - 6)
      .attr("x2", w)
      .attr("y1", yAdv)
      .attr("y2", yAdv)
      .attr("stroke", t.border);

    const rewardG = svg.append("g");
    const advG = svg.append("g");
    // One status line over the chart: the spread, then what it means.
    const note = svg.append("text").attr("x", 0).attr("y", noteY);
    const noteStd = note.append("tspan").attr("class", "fig-label").attr("font-weight", 650);
    const noteSay = note.append("tspan").attr("class", "fig-axis").attr("dx", 10);

    update = (k: number) => {
      const rewards: number[] = Array.from({ length: G }, (_, i) =>
        i < k ? 1 : 0
      );
      const mean = rewards.reduce((a: number, b: number) => a + b, 0) / G;
      const variance =
        rewards.reduce((a: number, r: number) => a + (r - mean) ** 2, 0) / G;
      const std = Math.sqrt(variance);
      const adv: number[] = rewards.map(r => (std === 0 ? 0 : (r - mean) / std));

      rewardG
        .selectAll<SVGCircleElement, number>("circle")
        .data(rewards)
        .join("circle")
        .attr("cx", (_d: number, i: number) => x(i))
        .attr("cy", yReward)
        .attr("r", half)
        .attr("fill", (r: number) => (r === 1 ? t.accent : t.surface))
        .attr("stroke", (r: number) => (r === 1 ? t.accent : FAIL))
        .attr("stroke-width", 1.5);

      advG
        .selectAll<SVGRectElement, number>("rect")
        .data(adv)
        .join("rect")
        .attr("x", (_d: number, i: number) => x(i) - half)
        .attr("width", 2 * half)
        .attr("rx", 3)
        .attr("y", (a: number) => (a >= 0 ? yAdv - a * SCALE : yAdv))
        .attr("height", (a: number) =>
          Math.max(Math.abs(a) * SCALE, a === 0 ? 2 : 0)
        )
        .attr("fill", (a: number) =>
          a === 0 ? "#c9d7e2" : a > 0 ? t.accent : "#e08a5a"
        );

      const dead = std === 0;
      noteStd.text(`Reward std ${std.toFixed(2)}`);
      noteSay
        .text(
          dead
            ? `Every advantage is 0 · ${k === 0 ? "too hard" : "too easy"}`
            : "Gradient is non-zero"
        )
        .style("fill", dead ? t.fg : null);
    };
    update(Number(input.value));
  });

  function draw(k: number) {
    readout.textContent = `${k} of ${G}`;
    update(k);
    const rewards = Array.from({ length: G }, (_, i) => (i < k ? 1 : 0));
    const mean = k / G;
    const std = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / G);
    const dead = std === 0;
    // What the chart shows, spoken with the slider value.
    const advText = dead
      ? "every advantage is 0, no gradient"
      : `solved rollouts get advantage ${((1 - mean) / std).toFixed(2)}, failed ones ${((0 - mean) / std).toFixed(2)}`;
    input.setAttribute("aria-valuetext", `${k} of ${G} correct: reward std ${std.toFixed(2)}, ${advText}`);
  }
  input.addEventListener("input", () => draw(Number(input.value)));
  draw(3);

  legend(node, [
    { label: "Solved (reward 1)", swatch: t.accent },
    { label: "Failed (reward 0)", swatch: FAIL },
    { label: "Negative advantage", swatch: "#e08a5a" },
  ]);
  caption(
    node,
    "Not a measurement: the GRPO advantage definition evaluated over one group of 8 rollouts. " +
      "Drag the slider to 0 or to 8. The rewards differ completely, the advantages are identical, and both produce no gradient. " +
      "That is why 'too easy' and 'too hard' look the same from outside the reward function."
  );
}

/* ---------------------------------------------------------------- dispatch */

const FIGURES: Record<string, (node: Element) => void | Promise<void>> = {
  "induction-heatmap": inductionHeatmap,
  "induction-ranked": inductionRanked,
  "gpt-parameters": gptParameters,
  "grpo-advantage": grpoAdvantage,
};

function render() {
  document.querySelectorAll<HTMLElement>("[data-figure]").forEach(node => {
    if (node.dataset.rendered) return;
    const name = node.dataset.figure ?? "";
    const fn = name.startsWith("venture:")
      ? (n: Element) => ventureFigure(n, name.slice("venture:".length))
      : name.startsWith("chart:")
        ? (n: Element) => specFigure(n, name.slice("chart:".length))
        : FIGURES[name];
    if (!fn) return;
    node.dataset.rendered = "1";
    node.classList.add("fig");
    Promise.resolve(fn(node))
      .then(() => {
        if (node.dataset.motion === "self") return; // diagrams animate themselves
        animateIn(node);
        interactive(node);
        hoverHint(node);
      })
      .catch(err => {
        // Leave the post readable: every figure restates numbers the prose already has.
        node.dataset.rendered = "";
        console.error(`[figure:${name}]`, err);
      });
  });
}

document.addEventListener("astro:page-load", render);
render();
