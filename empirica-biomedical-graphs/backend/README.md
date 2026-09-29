# Empirica - Backend

FastAPI backend for processing biomedical PDFs and generating knowledge graphs.

## Features

- 📄 **PDF Processing**: Extract text from biomedical PDFs using PyMuPDF
- 🧬 **Named Entity Recognition**: scispaCy-powered entity extraction (genes, chemicals, diseases, etc.)
- 🔗 **Relationship Extraction**: Co-occurrence and pattern-based relationship detection
- 🤖 **LLM Enhancement** (Optional): Semantic relationship classification using Anthropic Claude
- 📊 **Graph Analytics**: Community detection, centrality measures, and statistics
- 💾 **Persistence**: SQLite/PostgreSQL for saving projects

## Setup

### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download scispaCy Model

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings (optional: add LLM API keys)
```

### 4. Run Server

```bash
python -m app.main
# Or: uvicorn app.main:app --reload
```

Server runs on `http://localhost:8000`

## API Endpoints

### `POST /api/process`
Upload PDFs and generate knowledge graph

**Request:**
- `files`: List of PDF files
- `project_name` (optional): Project name
- `enable_llm` (optional): Enable LLM-powered extraction

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 0.5,
  "message": "Extracting entities...",
  "result": null
}
```

### `GET /api/status/{job_id}`
Check processing status

### `POST /api/graph/filter`
Filter graph by criteria (degree, entity types, top-N)

### `POST /api/analytics`
Compute graph analytics

### `GET /api/projects`
List saved projects

## Architecture

```
backend/
├── app/
│   ├── main.py                 # FastAPI app & endpoints
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── schemas.py          # Pydantic models
│   │   └── database.py         # SQLAlchemy models
│   └── services/
│       ├── pdf_processor.py    # PDF extraction
│       ├── ner_service.py      # Named Entity Recognition
│       ├── relationship_extractor.py
│       ├── graph_builder.py    # Graph construction
│       └── llm_service.py      # LLM integration
├── requirements.txt
└── uploads/                    # Temporary PDF storage
```

## Technologies

- **FastAPI**: Modern async API framework
- **PyMuPDF**: High-performance PDF processing
- **scispaCy**: Biomedical NER models
- **NetworkX**: Graph algorithms
- **python-louvain**: Community detection
- **SQLAlchemy**: Database ORM
- **Anthropic** (optional): LLM enhancement

