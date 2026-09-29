# Measurements

`pipeline_measurements.json` was produced by `scripts/measure_pipeline.py` (added in this port; it calls the project's own PDFProcessor, NERService, RelationshipExtractor and GraphBuilder unchanged, no LLM or paid API).

Setup: Python 3.11 (uv venv) on an Apple-silicon Mac (CPU only), spacy 3.7.4, scispacy 0.5.5, en_ner_bionlp13cg_md 0.5.4, networkx 3.2.1, python-louvain 0.16, PyMuPDF 1.24.13, numpy 1.26.4.

```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python spacy==3.7.4 scispacy==0.5.5 networkx==3.2.1 python-louvain==0.16 PyMuPDF==1.24.13 numpy==1.26.4 pydantic==2.9.2 pydantic-settings==2.6.1 anthropic==0.39.0 httpx==0.25.2 requests beautifulsoup4 lxml sqlalchemy==2.0.23 python-dotenv aiofiles fastapi==0.115.0 python-jose passlib
uv pip install --python .venv/bin/python https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
python scripts/measure_pipeline.py PMC13028654 PMC13072840 PMC13090758
```

Inputs: three Europe PMC open-access papers (CC BY 4.0; PMC13090758's XML has no license tag but Europe PMC's metadata lists "cc by"). Publisher PDFs sat behind a bot check, so the script builds a plain-text PDF from Europe PMC full-text XML (abstract + body paragraphs, references excluded). The PDF layout is therefore not real-world PDF extraction (no columns, headers, footnotes), which likely makes sentence splitting cleaner than on real PDFs.

Caveats: single run, no repeats, n=3 papers, no ground truth, so nothing here measures accuracy. NER dominates wall time (about 7 to 13 s per paper). The top-centrality nodes include generic terms such as "cells", which shows the limits of the entity filtering. `edges_after_merge` exceeds `graph_edges` because GraphBuilder drops edges whose endpoints are not among the (min 4 occurrences) kept entities. The FastAPI app, frontend, and LLM/agentic/RAG paths were not run; tests do not exist in the repo.
