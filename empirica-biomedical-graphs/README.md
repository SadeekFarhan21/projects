> **Provenance and credit (added for this blog port).**
> Empirica (internal name "SynapseMapper") is a hackathon project built by a team, mainly by **Jalen Francis** (NLP/graph pipeline, RAG, LLM and agentic modules, Google OAuth backend, most of the frontend). **Farhan Sadeek** contributed login UI and branding, CORS/request-logging/health endpoints in `backend/app/main.py`, and deployment configs (not included here). **Edward Kim** wrote the first Google OAuth pass. Upstream: <https://github.com/jalenfran/synapsemapper>; team fork: <https://github.com/SadeekFarhan21/empirica> (snapshot at `4291745`, 2025-10-26).
> The README below is the upstream README, unmodified apart from this header. Some of its claims are not backed by the code: e.g. semantic embeddings / sentence-transformers are not implemented (`rag_service.py` has a TODO), and the timing figures in `RAG_SYSTEM.md` are unmeasured estimates. The README says MIT, but the upstream repo has no LICENSE file.
> **Edits made in this copy:** removed committed secrets (Lava key literals in `setup.sh`, a hardcoded Anthropic key in `backend/app/services/llm_service.py`, a Google client ID in `setup.sh`) and one hardcoded server IP in `config.py`; omitted `backend/.env.backup`, `frontend/OAUTH_SETUP.md`, `package-lock.json`. Because of this the LLM features will not run until you supply your own key. Added `scripts/measure_pipeline.py` and `results/` (see `results/README.md`).

# 🧬 Empirica

**Transform biomedical PDFs into interactive knowledge graphs with AI-powered insights**

Empirica is a sophisticated research intelligence tool that ingests biomedical PDF documents and automatically generates interactive, force-directed knowledge graphs. Each PDF gets its own graph, which can be dynamically combined for multi-document analysis. Enhanced with RAG (Retrieval-Augmented Generation) and LLM-powered insights via Lava Payments + Anthropic Claude.

![Version](https://img.shields.io/badge/version-2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![React](https://img.shields.io/badge/react-18.2-blue.svg)
![RAG](https://img.shields.io/badge/RAG-enabled-purple.svg)

## ✨ Features

### 🎯 Core Capabilities
- **Per-PDF Graph System** - Each PDF gets its own knowledge graph
- **Dynamic Graph Merging** - Combine multiple PDF graphs by selection/deselection
- **Named Entity Recognition** - scispaCy-powered identification of biomedical entities
- **Relationship Extraction** - Pattern-based and co-occurrence analysis
- **Interactive Visualization** - 2D/3D force-directed graphs with smooth physics
- **Advanced Analytics** - Community detection, centrality analysis, graph statistics

### 🤖 AI-Powered Features
- **RAG System** - Document chunking, semantic indexing, and context retrieval
- **Hypothesis Generation** - LLM-powered research insights from your documents
- **Conversational AI** - Chat with your knowledge graph using natural language
- **Evidence-Based Insights** - All AI responses grounded in your source documents
- **Lava Payments Integration** - Usage-based billing for AI API calls

### 🚀 Advanced Features
- **Smart PDF Management** - Add/remove PDFs from existing projects
- **PDF Selection** - Toggle PDFs to dynamically update the merged graph
- **Real-time Processing** - Live progress tracking with background job processing
- **Import/Export System** - Full project state including RAG indices
- **Multi-user Support** - OAuth authentication with Google
- **Persistent Storage** - SQLite database with per-PDF graph storage

### 🎨 Beautiful UI
- Modern gradient design with dark theme
- Responsive layout for all screen sizes
- Smooth animations and transitions
- Interactive tooltips with evidence sentences
- Entity color-coding by type

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   React + Vite  │  HTTP   │   FastAPI       │
│   TypeScript    │ ────▶   │   Python 3.9+   │
│   Tailwind CSS  │         │                  │
└─────────────────┘         └──────────────────┘
        │                            │
        │                            ▼
        │                   ┌──────────────────┐
        │                   │   NLP Pipeline   │
        │                   │  - PyMuPDF       │
        │                   │  - scispaCy      │
        │                   │  - NetworkX      │
        │                   └──────────────────┘
        │                            │
        ▼                            ▼
┌─────────────────┐         ┌──────────────────┐
│  Force Graph    │         │   Graph Builder  │
│  - 2D/3D Views  │ ◀────   │  - Communities   │
│  - Interactions │  JSON   │  - Centrality    │
└─────────────────┘         └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+**
- **Node.js 18+**
- **npm or yarn**

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd calhacks
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download scispaCy model (this may take a few minutes)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz

# Configure Lava Payments (required for AI features)
cat > .env << EOF
LAVA_SECRET_KEY=aks_live_...
LAVA_CONNECTION_SECRET=cons_live_...
LAVA_PRODUCT_SECRET=ps_live_...
ENABLE_LAVA=true
EOF

# Run backend server
uvicorn app.main:app --reload
```

Backend runs on `http://localhost:8000`

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on `http://localhost:5173`

### 4. Open Browser
Navigate to `http://localhost:5173` and start uploading PDFs!

## 📖 Usage

### Basic Workflow
1. **Upload PDFs** - Drag & drop biomedical PDF files
2. **Wait for Processing** - Monitor real-time progress (NER, relationship extraction, graph building, RAG indexing)
3. **Explore Graph** - Pan, zoom, click nodes to investigate
4. **Generate Hypotheses** - Click "Generate Hypotheses" for AI-powered insights
5. **Chat with Graph** - Ask questions about your documents in natural language
6. **Manage PDFs** - Add/remove PDFs or toggle selection to update the graph
7. **Export/Import** - Save full project state including RAG indices

### Advanced Features

#### Per-PDF Graphs
- Each PDF gets its own knowledge graph stored separately
- Select/deselect PDFs to dynamically merge graphs
- Add new PDFs to existing projects
- Remove PDFs and their associated graph data

#### RAG-Enhanced AI
- **Document Chunking**: Smart sentence-aware chunking with entity tracking
- **Semantic Search**: Find relevant content using embeddings
- **Graph-Aware Retrieval**: Combine semantic similarity with graph connectivity
- **Evidence-Based**: All AI responses cite source documents

#### Graph Filtering
- **Entity Types**: Toggle specific biomedical entities
- **PDF Selection**: Show/hide graphs from specific PDFs
- **Min Degree**: Show only highly connected nodes
- **Search**: Find entities by name

#### View Modes
- **2D View**: High performance, ideal for large graphs
- **3D View**: Impressive visualization, better for presentations
- **Labels Toggle**: Show/hide node labels

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern async API framework
- **PyMuPDF** - High-performance PDF processing
- **scispaCy** - Biomedical NER (en_ner_bionlp13cg_md model)
- **NetworkX** - Graph algorithms and analysis
- **sentence-transformers** - RAG embeddings
- **python-louvain** - Community detection
- **SQLAlchemy** - Database ORM
- **Anthropic Claude** - LLM (via Lava Payments)
- **Lava Payments** - Usage-based AI billing

### Frontend
- **React 18** - UI library with hooks
- **TypeScript** - Type safety
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **react-force-graph** - WebGL-powered graph rendering
- **Recharts** - Analytics visualizations
- **Zustand** - Lightweight state management
- **Lucide React** - Modern icon library

## 📊 API Endpoints

### Projects
- `POST /api/projects` - Create project & upload PDFs
- `GET /api/projects` - List user's projects
- `GET /api/projects/{id}` - Get project details
- `GET /api/projects/{id}/pdfs` - List project PDFs
- `GET /api/projects/{id}/graph` - Get merged graph from selected PDFs
- `DELETE /api/projects/{id}` - Delete project

### PDFs
- `POST /api/projects/{id}/pdfs` - Add PDFs to existing project
- `DELETE /api/projects/{project_id}/pdfs/{pdf_id}` - Remove PDF from project
- `POST /api/projects/{id}/pdfs/selection` - Update PDF selection status

### AI Features (RAG-Enhanced)
- `POST /api/hypotheses` - Generate research insights
- `POST /api/chat` - Chat with knowledge graph

### Import/Export
- `POST /api/export` - Export project with RAG index
- `POST /api/import` - Import project and restore RAG index

### External Data
- `GET /api/pubmed/search` - Search PubMed
- `GET /api/clinicaltrials/search` - Search ClinicalTrials.gov

### Processing
- `GET /api/processing/{job_id}` - Check processing status

## 🎯 Use Cases

### Research Applications
- **Literature Review** - Discover connections across multiple papers
- **Hypothesis Generation** - Find unexpected relationships
- **Drug Discovery** - Identify drug-disease-gene interactions
- **Biomarker Discovery** - Explore disease-protein associations

### Hackathon Tips
- **Demo Impact** - 3D visualization is visually impressive for judges
- **Sample Data** - Prepare biomedical PDFs beforehand
- **Performance** - Start with 2-5 PDFs for demos
- **Analytics** - Showcase community detection and centrality
- **Export** - Show data portability with exports

## 🧪 Development

### Backend Development
```bash
cd backend
source venv/bin/activate

# Run with auto-reload
python -m app.main

# Or use uvicorn directly
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend

# Dev server with hot reload
npm run dev

# Type checking
npm run build

# Linting
npm run lint
```

### Project Structure
```
calhacks/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app
│   │   ├── config.py                    # Configuration
│   │   ├── models/                      # Data models
│   │   │   ├── schemas.py               # Pydantic models
│   │   │   └── database.py              # SQLAlchemy models (per-PDF graphs)
│   │   └── services/                    # Business logic
│   │       ├── pdf_processor.py         # PDF text extraction
│   │       ├── ner_service.py           # Named entity recognition
│   │       ├── relationship_extractor.py # Entity relationships
│   │       ├── graph_builder.py         # NetworkX graph construction
│   │       ├── document_chunker.py      # RAG document chunking
│   │       ├── rag_service.py           # RAG indexing & retrieval
│   │       ├── content_insight_agent.py # Insight generation
│   │       ├── graph_agent.py           # Conversational AI
│   │       ├── llm_service.py           # LLM abstraction
│   │       ├── lava_service.py          # Lava Payments integration
│   │       ├── pubmed_service.py        # PubMed API
│   │       └── ctgov_service.py         # ClinicalTrials.gov API
│   ├── uploads/                         # PDF storage & RAG indices
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/                  # React components
│   │   │   ├── Sidebar.tsx              # Main navigation
│   │   │   ├── PDFSelector.tsx          # PDF management
│   │   │   ├── ForceGraph2DView.tsx     # 2D visualization
│   │   │   ├── ForceGraph3DView.tsx     # 3D visualization
│   │   │   ├── ChatPanel.tsx            # AI chat
│   │   │   └── ...
│   │   ├── services/                    # API client
│   │   ├── store/                       # Zustand state management
│   │   ├── types/                       # TypeScript types
│   │   └── App.tsx
│   └── package.json
├── ARCHITECTURE.md                      # Detailed architecture docs
├── RAG_SYSTEM.md                        # RAG implementation details
├── LAVA_SETUP.md                        # Lava Payments setup guide
└── README.md                            # This file
```

## 🐛 Troubleshooting

### Backend Issues

**scispaCy model not found:**
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

**CORS errors:**
- Check `CORS_ORIGINS` in backend `.env`
- Ensure frontend URL is in allowed origins

**Slow processing:**
- Start with smaller PDFs
- Disable LLM extraction for faster processing
- Check system resources

### Frontend Issues

**Blank graph:**
- Check browser console for errors
- Verify backend is running and accessible
- Check API URL in frontend `.env`

**Performance issues:**
- Use 2D view for large graphs
- Apply filters to reduce node count
- Disable node labels when zoomed out

## 🚀 Deployment

### Backend (Docker)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend (Vercel/Netlify)
```bash
cd frontend
npm run build
# Deploy dist/ folder
```

## 🤝 Contributing

This is a hackathon project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use this project for your hackathon or research!

## 🙏 Acknowledgments

- **scispaCy** - Biomedical NLP models
- **NetworkX** - Graph algorithms
- **react-force-graph** - Graph visualization
- **FastAPI** - API framework

## 📧 Contact

Built for CalHacks 2025 🐻

---

**Happy Mapping! 🧬**

