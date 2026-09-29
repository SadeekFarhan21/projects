#!/usr/bin/env python3
"""Build clickable SVG versions of diagrams that have a links file.

For diagrams/<name>.links.json ({element id: {"href": "#section", "tip": "..."}}),
each listed box gets that link, the diagram is rendered to SVG with the site's
fonts, and every link gets a tooltip. The result goes to
source/img/diagrams/<name>.svg, which tools/embed-diagrams.py inlines in the post
in place of the WebP, so hovering a box highlights it and clicking it jumps to
the section that explains it.

  <excalidraw skill venv>/bin/python tools/interactive-diagrams.py
"""
import html
import json
import pathlib
import re
import subprocess
import sys
import tempfile

SITE = pathlib.Path(__file__).resolve().parent.parent
DIAGRAMS = SITE / "diagrams"
OUT = SITE / "source" / "img" / "diagrams"
RENDER = SITE / "tools" / "excalidraw" / "render_excalidraw.py"


def build(links_path: pathlib.Path) -> None:
    name = links_path.name[: -len(".links.json")]
    links = json.loads(links_path.read_text())
    data = json.loads((DIAGRAMS / f"{name}.excalidraw").read_text())
    ids = {e["id"] for e in data["elements"]}
    missing = sorted(set(links) - ids)
    if missing:
        sys.exit(f"{name}: no elements with ids {missing}")
    for el in data["elements"]:
        if el["id"] in links:
            el["link"] = links[el["id"]]["href"]
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / f"{name}.excalidraw"
        src.write_text(json.dumps(data))
        svg_path = pathlib.Path(tmp) / f"{name}.svg"
        subprocess.run([sys.executable, str(RENDER), str(src), "--output", str(pathlib.Path(tmp) / "x.png"),
                        "--scale", "1", "--svg", str(svg_path)], check=True, capture_output=True)
        svg = svg_path.read_text()
    # The page already loads Quicksand and Fragment Mono; drop Excalidraw's embedded fonts.
    svg = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", svg)
    tips = {v["href"]: v["tip"] for v in links.values()}

    def titled(m: re.Match) -> str:
        href = html.unescape(m.group(1))
        tip = tips.get(href, "")
        return f'<a href="{m.group(1)}" class="excal-hot"><title>{html.escape(tip)}</title>'

    svg = re.sub(r'<a href="([^"]+)">', titled, svg)
    svg = svg.replace("<svg ", '<svg class="excal-svg" role="img" ', 1)
    # One line, so the Markdown renderer keeps it as a single HTML block.
    svg = re.sub(r">\s+<", "><", svg).replace("\n", " ")
    (OUT / f"{name}.svg").write_text(svg)
    print(f"{name}: {len(links)} links, {len(svg) // 1024} KB")


def main() -> int:
    for links_path in sorted(DIAGRAMS.glob("*.links.json")):
        build(links_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
