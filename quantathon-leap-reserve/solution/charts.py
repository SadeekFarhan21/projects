from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .assumptions import YEARS


def _currency(value: float) -> str:
    return f"${value/1_000_000:.1f}M"


def _write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _chart_shell(title: str, width: int = 900, height: int = 520) -> Tuple[int, int, int, int, int]:
    left = 80
    right = 30
    top = 50
    bottom = 70
    return width, height, left, right, top, bottom


def write_line_bar_chart(path: Path, title: str, bars: Dict[int, float], line: Dict[int, float], bar_label: str, line_label: str) -> None:
    width, height, left, right, top, bottom = _chart_shell(title)
    inner_width = width - left - right
    inner_height = height - top - bottom
    max_value = max(max(bars.values()), max(line.values())) * 1.1 or 1.0
    bar_width = inner_width / len(YEARS) * 0.6
    points = []
    bar_elems = []
    grid_elems = []
    label_elems = []

    for step in range(6):
        value = max_value * step / 5.0
        y = top + inner_height - (value / max_value) * inner_height
        grid_elems.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d8dde6' stroke-width='1' />")
        label_elems.append(f"<text x='18' y='{y+5:.1f}' font-size='12' fill='#334155'>{_currency(value)}</text>")

    for index, year in enumerate(YEARS):
        center_x = left + (index + 0.5) * inner_width / len(YEARS)
        bar_height = (bars[year] / max_value) * inner_height
        bar_y = top + inner_height - bar_height
        bar_elems.append(
            f"<rect x='{center_x - bar_width / 2:.1f}' y='{bar_y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='#0f766e' rx='4' />"
        )
        line_y = top + inner_height - (line[year] / max_value) * inner_height
        points.append(f"{center_x:.1f},{line_y:.1f}")
        label_elems.append(f"<text x='{center_x:.1f}' y='{height-25}' text-anchor='middle' font-size='11' fill='#334155'>{year}</text>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='24' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>{title}</text>
{''.join(grid_elems)}
<polyline fill='none' stroke='#b91c1c' stroke-width='3' points='{' '.join(points)}' />
{''.join(bar_elems)}
{''.join(label_elems)}
<line x1='{left}' y1='{top+inner_height}' x2='{width-right}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<rect x='{width-250}' y='18' width='14' height='14' fill='#0f766e' rx='3' />
<text x='{width-230}' y='30' font-size='12' fill='#334155'>{bar_label}</text>
<line x1='{width-150}' y1='25' x2='{width-120}' y2='25' stroke='#b91c1c' stroke-width='3' />
<text x='{width-110}' y='30' font-size='12' fill='#334155'>{line_label}</text>
</svg>"""
    _write_svg(path, svg)


def write_histogram(path: Path, title: str, values: Sequence[float], bucket_count: int = 16) -> None:
    width, height, left, right, top, bottom = _chart_shell(title)
    inner_width = width - left - right
    inner_height = height - top - bottom
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    bins = [0 for _ in range(bucket_count)]
    for value in values:
        index = min(bucket_count - 1, int((value - minimum) / span * bucket_count))
        bins[index] += 1
    max_count = max(bins) or 1
    bar_width = inner_width / bucket_count * 0.8
    elements = []
    labels = []
    for step in range(6):
        count = max_count * step / 5.0
        y = top + inner_height - (count / max_count) * inner_height
        elements.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d8dde6' stroke-width='1' />")
        labels.append(f"<text x='18' y='{y+5:.1f}' font-size='12' fill='#334155'>{int(count)}</text>")
    for index, count in enumerate(bins):
        center_x = left + (index + 0.5) * inner_width / bucket_count
        bar_height = (count / max_count) * inner_height
        bar_y = top + inner_height - bar_height
        elements.append(
            f"<rect x='{center_x - bar_width / 2:.1f}' y='{bar_y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='#1d4ed8' rx='4' />"
        )
        bucket_value = minimum + span * index / bucket_count
        labels.append(f"<text x='{center_x:.1f}' y='{height-25}' text-anchor='middle' font-size='10' fill='#334155'>{_currency(bucket_value)}</text>")
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='24' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>{title}</text>
{''.join(elements)}
{''.join(labels)}
<line x1='{left}' y1='{top+inner_height}' x2='{width-right}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
</svg>"""
    _write_svg(path, svg)


def write_two_line_band_chart(
    path: Path,
    title: str,
    median_a: Dict[int, float],
    low_a: Dict[int, float],
    high_a: Dict[int, float],
    median_b: Dict[int, float],
    low_b: Dict[int, float],
    high_b: Dict[int, float],
    label_a: str,
    label_b: str,
) -> None:
    width, height, left, right, top, bottom = _chart_shell(title)
    inner_width = width - left - right
    inner_height = height - top - bottom
    combined = list(low_a.values()) + list(high_a.values()) + list(low_b.values()) + list(high_b.values())
    minimum = min(combined)
    maximum = max(combined)
    span = (maximum - minimum) or 1.0

    def to_point(year: int, value: float) -> str:
        index = YEARS.index(year)
        x = left + (index + 0.5) * inner_width / len(YEARS)
        y = top + inner_height - ((value - minimum) / span) * inner_height
        return f"{x:.1f},{y:.1f}"

    def band_polygon(low: Dict[int, float], high: Dict[int, float]) -> str:
        top_points = [to_point(year, high[year]) for year in YEARS]
        bottom_points = [to_point(year, low[year]) for year in reversed(YEARS)]
        return " ".join(top_points + bottom_points)

    grid = []
    labels = []
    for step in range(6):
        value = minimum + span * step / 5.0
        y = top + inner_height - ((value - minimum) / span) * inner_height
        grid.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d8dde6' stroke-width='1' />")
        labels.append(f"<text x='18' y='{y+5:.1f}' font-size='12' fill='#334155'>{_currency(value)}</text>")
    year_labels = [
        f"<text x='{left + (index + 0.5) * inner_width / len(YEARS):.1f}' y='{height-25}' text-anchor='middle' font-size='11' fill='#334155'>{year}</text>"
        for index, year in enumerate(YEARS)
    ]
    median_points_a = " ".join(to_point(year, median_a[year]) for year in YEARS)
    median_points_b = " ".join(to_point(year, median_b[year]) for year in YEARS)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='24' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>{title}</text>
{''.join(grid)}
<polygon points='{band_polygon(low_a, high_a)}' fill='#0f766e' opacity='0.18' />
<polygon points='{band_polygon(low_b, high_b)}' fill='#b45309' opacity='0.18' />
<polyline points='{median_points_a}' fill='none' stroke='#0f766e' stroke-width='3' />
<polyline points='{median_points_b}' fill='none' stroke='#b45309' stroke-width='3' />
{''.join(labels)}
{''.join(year_labels)}
<line x1='{left}' y1='{top+inner_height}' x2='{width-right}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{width-240}' y1='25' x2='{width-210}' y2='25' stroke='#0f766e' stroke-width='3' />
<text x='{width-200}' y='30' font-size='12' fill='#334155'>{label_a}</text>
<line x1='{width-120}' y1='25' x2='{width-90}' y2='25' stroke='#b45309' stroke-width='3' />
<text x='{width-80}' y='30' font-size='12' fill='#334155'>{label_b}</text>
</svg>"""
    _write_svg(path, svg)


def write_breakeven_chart(path: Path, breakeven_data: dict) -> None:
    """Line chart showing p95 funding vs trigger rate with threshold line."""
    sweep = breakeven_data["sweep_data"]
    funding_level = breakeven_data.get("funding_level")
    width, height, left, right, top, bottom = _chart_shell("Break-Even Analysis: Uptake Rate vs P95 Funding")
    left = 100  # wider for dollar labels
    inner_width = width - left - right
    inner_height = height - top - bottom

    rates = [d["trigger_rate"] for d in sweep]
    p95s = [d["p95"] for d in sweep]
    min_p95 = min(p95s) * 0.9
    max_p95 = max(p95s) * 1.1
    span = max_p95 - min_p95 or 1.0

    def to_point(i, val):
        x = left + (i + 0.5) * inner_width / len(sweep)
        y = top + inner_height - ((val - min_p95) / span) * inner_height
        return f"{x:.1f},{y:.1f}"

    points = " ".join(to_point(i, p) for i, p in enumerate(p95s))

    grid = []
    labels = []
    for step in range(6):
        val = min_p95 + span * step / 5.0
        y = top + inner_height - ((val - min_p95) / span) * inner_height
        grid.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d8dde6' stroke-width='1' />")
        labels.append(f"<text x='14' y='{y+5:.1f}' font-size='12' fill='#334155'>{_currency(val)}</text>")

    # X-axis labels (every 4th rate)
    for i, r in enumerate(rates):
        if i % 4 == 0 or i == len(rates) - 1:
            cx = left + (i + 0.5) * inner_width / len(sweep)
            labels.append(f"<text x='{cx:.1f}' y='{height-25}' text-anchor='middle' font-size='11' fill='#334155'>{r:.4f}</text>")

    # Threshold line
    threshold_elem = ""
    if funding_level and min_p95 <= funding_level <= max_p95:
        ty = top + inner_height - ((funding_level - min_p95) / span) * inner_height
        threshold_elem = f"<line x1='{left}' y1='{ty:.1f}' x2='{width-right}' y2='{ty:.1f}' stroke='#b91c1c' stroke-width='2' stroke-dasharray='6,4' />"
        threshold_elem += f"\n<text x='{width-right-4}' y='{ty-6:.1f}' text-anchor='end' font-size='11' fill='#b91c1c'>Funding Level {_currency(funding_level)}</text>"

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='20' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>Break-Even: Uptake Rate vs P95 Funding</text>
{''.join(grid)}
{threshold_elem}
<polyline points='{points}' fill='none' stroke='#1d4ed8' stroke-width='3' />
{''.join(labels)}
<line x1='{left}' y1='{top+inner_height}' x2='{width-right}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<text x='{width/2}' y='{height-5}' text-anchor='middle' font-size='12' fill='#64748b'>Annual Trigger Start Rate</text>
</svg>"""
    _write_svg(path, svg)


def write_geographic_chart(path: Path, geographic_data: dict) -> None:
    """Horizontal bar chart of composite risk scores by zip code."""
    zips = geographic_data["zip_metrics"][:12]  # Top 12 zip codes
    width = 900
    height = max(400, 50 * len(zips) + 120)
    left, right, top, bottom = 120, 40, 50, 40
    inner_w = width - left - right
    inner_h = height - top - bottom

    max_score = max(z["composite_risk_score"] for z in zips) * 1.1 or 1.0
    bar_h = max(16, inner_h / len(zips) * 0.6)
    colors = ["#b91c1c", "#dc2626", "#ef4444", "#f87171", "#fca5a5", "#fecaca",
              "#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"]

    elems = []
    elems.append(f"<rect width='100%' height='100%' fill='white'/>")
    elems.append(f"<text x='{left}' y='28' font-size='20' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>Geographic Risk Scores by Zip Code</text>")

    for i, z in enumerate(zips):
        cy = top + (i + 0.5) * inner_h / len(zips)
        bar_w = (z["composite_risk_score"] / max_score) * inner_w
        color = colors[i % len(colors)]
        elems.append(f"<rect x='{left}' y='{cy - bar_h/2:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='{color}' rx='4' />")
        elems.append(f"<text x='{left-8}' y='{cy+5:.1f}' text-anchor='end' font-size='12' fill='#334155'>{z['zip_code']}</text>")
        elems.append(f"<text x='{left + bar_w + 6:.1f}' y='{cy+4:.1f}' font-size='11' fill='#64748b'>{z['composite_risk_score']:.3f}</text>")

    elems.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#475569' stroke-width='1.5' />")

    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n" + "\n".join(elems) + "\n</svg>"
    _write_svg(path, svg)


def write_comparison_bar_chart(path: Path, title: str, labels_values: List[Tuple[str, float]]) -> None:
    width, height, left, right, top, bottom = _chart_shell(title)
    inner_width = width - left - right
    inner_height = height - top - bottom
    max_value = max(value for _, value in labels_values) * 1.15 or 1.0
    bar_width = inner_width / len(labels_values) * 0.55
    colors = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed", "#be185d"]
    grids = []
    labels = []
    bars = []
    for step in range(6):
        value = max_value * step / 5.0
        y = top + inner_height - (value / max_value) * inner_height
        grids.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d8dde6' stroke-width='1' />")
        labels.append(f"<text x='18' y='{y+5:.1f}' font-size='12' fill='#334155'>{_currency(value)}</text>")
    for index, (label, value) in enumerate(labels_values):
        center_x = left + (index + 0.5) * inner_width / len(labels_values)
        bar_height = (value / max_value) * inner_height
        bar_y = top + inner_height - bar_height
        color = colors[index % len(colors)]
        bars.append(f"<rect x='{center_x - bar_width / 2:.1f}' y='{bar_y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' fill='{color}' rx='6' />")
        bars.append(f"<text x='{center_x:.1f}' y='{bar_y - 8:.1f}' text-anchor='middle' font-size='12' fill='#0f172a'>{_currency(value)}</text>")
        labels.append(f"<text x='{center_x:.1f}' y='{height-25}' text-anchor='middle' font-size='12' fill='#334155'>{label}</text>")
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='24' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>{title}</text>
{''.join(grids)}
{''.join(labels)}
{''.join(bars)}
<line x1='{left}' y1='{top+inner_height}' x2='{width-right}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_height}' stroke='#475569' stroke-width='1.5' />
</svg>"""
    _write_svg(path, svg)
