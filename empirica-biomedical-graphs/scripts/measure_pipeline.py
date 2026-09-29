"""Measure the Empirica core (non-LLM) pipeline on open-access papers.

Added by the blog port (not part of the upstream project). It imports the
project's own PDFProcessor / NERService / RelationshipExtractor / GraphBuilder
unchanged and reports counts and wall time. No network calls to LLMs.

Input papers are fetched from Europe PMC as full-text XML (CC BY only), turned
into a plain-text PDF with PyMuPDF (publisher PDFs were behind a bot check),
then fed through the pipeline exactly as backend/app/main.py does.

Usage (from repo root, inside a venv with backend/requirements + scispaCy model):
    python scripts/measure_pipeline.py PMC13028654 PMC13072840 PMC13090758
"""
import io, json, sys, time, textwrap, contextlib, urllib.request
from pathlib import Path
from lxml import etree
import fitz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

def fetch_xml(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    return urllib.request.urlopen(url, timeout=60).read()

def xml_to_text(xml):
    root = etree.fromstring(xml)
    lic = " ".join(root.xpath("//license//@*[local-name()='href']"))
    title = " ".join(root.xpath("//article-title")[0].itertext())
    paras = [" ".join("".join(p.itertext()).split()) for p in root.xpath("//abstract//p | //body//p")]
    return title, lic, "\n".join([title] + paras)

def text_to_pdf(text, path):
    doc = fitz.open()
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, 95) + [""]
    for i in range(0, len(lines), 60):
        page = doc.new_page()
        y = 50
        for ln in lines[i:i+60]:
            page.insert_text((40, y), ln, fontsize=9); y += 12
    doc.save(path)

def main(pmcids, workdir):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    from app.services import PDFProcessor, NERService, RelationshipExtractor, GraphBuilder
    pdfp, ner, rx = PDFProcessor(), NERService(), RelationshipExtractor()
    out = []
    for pid in pmcids:
        title, lic, text = xml_to_text(fetch_xml(pid))
        pdf = workdir / f"{pid}.pdf"; text_to_pdf(text, str(pdf))
        t0 = time.perf_counter()
        d = pdfp.process_pdfs([str(pdf)])[0]
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):  # NER prints debug output
            raw = ner.extract_entities_from_sentences(d["sentences"])
        t2 = time.perf_counter()
        filt = ner.filter_entities(raw)
        uniq = ner.get_unique_entities(filt)
        co = rx.extract_cooccurrence_relationships(filt)
        pat = rx.extract_pattern_relationships(filt)
        rels = rx.extract_all_relationships(filt)
        gb = GraphBuilder(); g = gb.build_graph(uniq, rels); an = gb.compute_analytics()
        t3 = time.perf_counter()
        types = {}
        for r in rels: types[r["relationship_type"]] = types.get(r["relationship_type"], 0) + 1
        top = sorted(an.centrality_scores.items(), key=lambda x: -x[1])[:5]
        top_deg = sorted(gb.graph.degree(weight=None), key=lambda x: -x[1])[:5]
        out.append({
            "pmcid": pid, "title": title, "license_href": lic,
            "pdf_pages": len(fitz.open(pdf)), "chars": d["char_count"],
            "sentences": d["sentence_count"],
            "sentences_with_entities_raw": len(raw),
            "sentences_with_entities_after_min4_filter": len(filt),
            "unique_entities": len(uniq),
            "cooccurrence_edges": len(co), "pattern_edges": len(pat),
            "edges_after_merge": len(rels), "edge_types_after_merge": types,
            "graph_nodes": an.total_nodes, "graph_edges": an.total_edges,
            "density": round(an.density, 4), "avg_degree": round(an.avg_degree, 2),
            "communities": len(an.communities),
            "largest_community": max((len(c) for c in an.communities), default=0),
            "entity_type_counts": an.entity_counts,
            "top5_betweenness": [(n, round(s, 4)) for n, s in top],
            "top5_degree": top_deg,
            "seconds_pdf_text_and_sentences": round(t1 - t0, 3),
            "seconds_ner": round(t2 - t1, 3),
            "seconds_filter_rel_graph_analytics": round(t3 - t2, 3),
        })
        print(pid, out[-1]["graph_nodes"], out[-1]["graph_edges"], out[-1]["seconds_ner"], flush=True)
    res = ROOT / "results" / "pipeline_measurements.json"
    res.write_text(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import tempfile; main(sys.argv[1:], Path(tempfile.gettempdir()) / "empirica_work")
