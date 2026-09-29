# Empirica Architecture

**Clean, modular biomedical research assistant with RAG-enhanced knowledge graphs**

## 🏗️ System Overview

Empirica is a full-stack application that processes biomedical research papers, extracts knowledge graphs, and provides AI-powered insights through a RAG (Retrieval-Augmented Generation) system.

### Core Features
- ✅ **Per-PDF Graph System**: Each PDF gets its own knowledge graph
- ✅ **Dynamic Graph Merging**: Combine multiple PDF graphs by selection
- ✅ **RAG Integration**: Document chunking, indexing, and semantic retrieval
- ✅ **LLM-Powered Insights**: Hypothesis generation and conversational AI via Lava + Anthropic Claude
- ✅ **Import/Export**: Full project state including RAG indices
- ✅ **Multi-user Support**: OAuth authentication with Google

---

## 📁 Project Structure

```
calhacks/
├── backend/                    # FastAPI Python backend
│   ├── app/
│   │   ├── config.py          # Configuration (Lava keys, etc.)
│   │   ├── main.py            # API endpoints
│   │   ├── models/
│   │   │   ├── database.py    # SQLAlchemy models (per-PDF graphs)
│   │   │   └── schemas.py     # Pydantic request/response schemas
│   │   └── services/
│   │       ├── pdf_processor.py           # PDF text extraction
│   │       ├── ner_service.py             # Named Entity Recognition
│   │       ├── relationship_extractor.py  # Entity relationship extraction
│   │       ├── graph_builder.py           # NetworkX graph construction
│   │       ├── document_chunker.py        # Smart PDF chunking for RAG
│   │       ├── rag_service.py             # RAG indexing & retrieval
│   │       ├── content_insight_agent.py   # Insight generation
│   │       ├── graph_agent.py             # Conversational AI
│   │       ├── llm_service.py             # LLM abstraction layer
│   │       ├── lava_service.py            # Lava Payments integration
│   │       ├── pubmed_service.py          # PubMed API integration
│   │       └── ctgov_service.py           # ClinicalTrials.gov API
│   ├── uploads/               # PDF storage & RAG indices
│   ├── synapse_mapper.db      # SQLite database
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables (Lava keys)
│
├── frontend/                  # React + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx           # Main app component
│   │   ├── components/
│   │   │   ├── Sidebar.tsx            # Main navigation & controls
│   │   │   ├── UploadPanel.tsx        # PDF upload interface
│   │   │   ├── ProjectSelection.tsx   # Project management
│   │   │   ├── PDFSelector.tsx        # PDF selection & management
│   │   │   ├── ForceGraph2DView.tsx   # 2D graph visualization
│   │   │   ├── ForceGraph3DView.tsx   # 3D graph visualization
│   │   │   ├── NodeDetails.tsx        # Entity details panel
│   │   │   ├── Analytics.tsx          # Graph analytics
│   │   │   ├── ChatPanel.tsx          # AI chat interface
│   │   │   └── ExportMenu.tsx         # Import/export controls
│   │   ├── services/
│   │   │   └── api.ts        # Backend API client
│   │   ├── store/
│   │   │   └── useStore.ts   # Zustand global state
│   │   └── types/
│   │       └── index.ts      # TypeScript type definitions
│   └── package.json          # Node dependencies
│
└── Documentation
    ├── README.md             # Main project documentation
    ├── ARCHITECTURE.md       # This file
    ├── RAG_SYSTEM.md         # RAG implementation details
    └── LAVA_SETUP.md         # Lava Payments setup guide
```

---

## 🔄 Data Flow

### 1. PDF Upload & Processing

```
User uploads PDFs
    ↓
FastAPI receives files → Saves to uploads/
    ↓
Background task (process_pdfs_background):
    For each PDF:
        1. Extract text (PDFProcessor)
        2. Extract entities (NERService)
        3. Chunk document (DocumentChunker) ← RAG
        4. Index chunks (RAGService) ← RAG
        5. Extract relationships (RelationshipExtractor)
        6. Build graph (GraphBuilder)
        7. Save to database (PDFGraphNode, PDFGraphEdge)
    ↓
Save RAG index to uploads/{project_id}_rag_index.pkl
    ↓
Return project_id to frontend
```

### 2. Graph Visualization

```
User selects project
    ↓
Frontend calls /api/projects/{id}/graph
    ↓
Backend:
    1. Query selected PDFs
    2. Load PDFGraphNodes & PDFGraphEdges
    3. Merge graphs (combine nodes, aggregate edges)
    4. Return merged graph JSON
    ↓
Frontend renders with ForceGraph2D/3D
```

### 3. Hypothesis Generation (RAG-Enhanced)

```
User clicks "Generate Hypotheses"
    ↓
Frontend sends graph + project_id to /api/hypotheses
    ↓
Backend:
    1. Build NetworkX graph from nodes/edges
    2. Load RAG index (uploads/{project_id}_rag_index.pkl)
    3. Set graph context in RAGService
    4. Retrieve top entities from graph
    5. RAG retrieval: semantic search + graph context
    6. Build RAG-enhanced prompt
    7. Send to LLM (via LavaService → Anthropic Claude)
    8. Return insights
    ↓
Frontend displays in sidebar
```

### 4. Chat (RAG-Enhanced)

```
User types message in chat
    ↓
Frontend sends message + graph + project_id to /api/chat
    ↓
Backend:
    1. Load RAG index
    2. Extract entities from user message
    3. RAG retrieval: query + entities + graph context
    4. Build RAG-enhanced prompt
    5. Send to LLM via GraphConversationalAgent
    6. Return response with citations
    ↓
Frontend displays in ChatPanel
```

### 5. Import/Export

**Export:**
```
User clicks Export
    ↓
Backend:
    1. Fetch project metadata
    2. Fetch all PDFs with their graphs
    3. Load RAG index
    4. Serialize RAG index to base64
    5. Build JSON with:
        - Project info
        - PDF metadata
        - Individual PDF graphs
        - RAG index (in settings)
    ↓
Frontend downloads JSON file
```

**Import:**
```
User uploads JSON file
    ↓
Backend:
    1. Parse JSON
    2. Create new project
    3. For each PDF:
        - Copy PDF file
        - Create Document record
        - Create PDFGraphNodes & PDFGraphEdges
    4. Restore RAG index from base64
    5. Save RAG index to uploads/
    ↓
Return new project_id
```

---

## 🗄️ Database Schema

### Core Tables

**`projects`**
- `id` (PK): UUID
- `name`: Project name
- `description`: Optional description
- `user_id`: Owner (FK to users)
- `created_at`: Timestamp

**`documents`** (PDFs)
- `id` (PK): UUID
- `project_id` (FK): Parent project
- `filename`: Original filename for display
- `file_path`: Full path to PDF file
- `processed`: Status (-1=error, 0=pending, 1=done)
- `selected`: Boolean (1=included in merged graph)
- `original_name`: Original upload filename
- `created_at`: Timestamp

**`pdf_graph_nodes`** (Per-PDF entities)
- `id` (PK): Auto-increment
- `document_id` (FK): Parent PDF
- `entity_id`: Entity name (e.g., "BDNF")
- `entity_type`: Type (GENE, PROTEIN, DISEASE, etc.)
- `count`: Occurrence count in PDF
- `degree`: Graph centrality

**`pdf_graph_edges`** (Per-PDF relationships)
- `id` (PK): Auto-increment
- `document_id` (FK): Parent PDF
- `source`: Source entity ID
- `target`: Target entity ID
- `weight`: Relationship strength
- `relationship_type`: Type (CO_OCCURRENCE, REGULATES, etc.)
- `evidence`: JSON array of evidence sentences

**`users`** (OAuth)
- `id` (PK): UUID
- `email`: User email
- `name`: Display name
- `google_id`: Google OAuth ID
- `created_at`: Timestamp

---

## 🧠 RAG System Architecture

### Components

**1. DocumentChunker** (`document_chunker.py`)
- Sentence-aware chunking (respects sentence boundaries)
- Configurable chunk size (default: 500 chars) & overlap (default: 100 chars)
- Entity tracking: links entities to chunks they appear in
- Page number preservation

**2. RAGService** (`rag_service.py`)
- **Indexing**: Stores text chunks with embeddings (via sentence-transformers)
- **Graph Context**: Links chunks to knowledge graph entities
- **Hybrid Retrieval**: Combines semantic similarity + graph connectivity
- **Context Assembly**: Builds rich prompts with relevant chunks + graph relationships
- **Persistence**: Saves/loads indices with pickle

**3. Integration Points**
- **PDF Processing**: Chunks indexed during upload
- **Hypothesis Generation**: RAG retrieves context for LLM prompts
- **Chat**: RAG retrieves context for user queries
- **Import/Export**: RAG index serialized in project JSON

### RAG Workflow

```
Document → Chunk → Embed → Index
                              ↓
User Query → Semantic Search + Graph Filter → Top K Chunks
                                                    ↓
                                        Build Prompt with Context
                                                    ↓
                                                  LLM
```

**See `RAG_SYSTEM.md` for detailed implementation.**

---

## 🤖 LLM Integration

### Architecture

**LLMService** (`llm_service.py`)
- Abstraction layer for LLM providers
- Currently supports: **Anthropic Claude** (via Lava)
- Methods:
  - `extract_relationships()`: Extract entity relationships from text
  - `generate_insights()`: Generate research hypotheses
  - `chat()`: Conversational responses

**LavaService** (`lava_service.py`)
- Handles Lava Payments authentication & billing
- Forwards requests to Anthropic API
- Tracks usage via Lava metadata
- Base64-encoded auth token with:
  - `secret_key`: Lava API key
  - `connection_secret`: Lava connection
  - `product_secret`: Lava product

### Configuration

**Environment Variables** (`.env`):
```bash
LAVA_SECRET_KEY=aks_live_...
LAVA_CONNECTION_SECRET=cons_live_...
LAVA_PRODUCT_SECRET=ps_live_...
ENABLE_LAVA=true
```

**See `LAVA_SETUP.md` for setup instructions.**

---

## 🎨 Frontend Architecture

### State Management (Zustand)

**`useStore.ts`** - Global state:
```typescript
{
  // Graph data
  graphData: { nodes: [], edges: [] },
  selectedNode: Node | null,
  
  // Project management
  currentProject: ProjectInfo | null,
  pdfs: PDFMetadata[],
  selectedPdfIds: Set<string>,
  
  // UI state
  viewMode: '2d' | '3d',
  hypotheses: Hypothesis[],
  chatMessages: Message[],
  
  // Actions
  setGraphData, selectNode, loadProject,
  updatePdfSelection, addPdfsToProject, ...
}
```

### Key Components

**Sidebar** - Main control panel
- Project selection
- PDF management (PDFSelector)
- Hypothesis generation
- Analytics
- Import/Export

**ForceGraph2D/3D** - Graph visualization
- Force-directed layout
- Node coloring by entity type
- Interactive zoom/pan
- Click to select nodes

**ChatPanel** - AI chat interface
- Conversational AI with graph context
- RAG-enhanced responses
- Citation support

**NodeDetails** - Entity information
- Entity type, occurrence count
- Connected entities
- Evidence sentences

---

## 🔌 API Endpoints

### Projects
- `POST /api/projects` - Create project & upload PDFs
- `GET /api/projects` - List user's projects
- `GET /api/projects/{id}` - Get project details
- `GET /api/projects/{id}/pdfs` - List project PDFs
- `GET /api/projects/{id}/graph` - Get merged graph
- `DELETE /api/projects/{id}` - Delete project

### PDFs
- `POST /api/projects/{id}/pdfs` - Add PDFs to project
- `DELETE /api/projects/{project_id}/pdfs/{pdf_id}` - Remove PDF
- `POST /api/projects/{id}/pdfs/selection` - Update PDF selection

### AI Features
- `POST /api/hypotheses` - Generate insights (RAG-enhanced)
- `POST /api/chat` - Chat with graph (RAG-enhanced)

### Import/Export
- `POST /api/export` - Export project JSON
- `POST /api/import` - Import project JSON

### External Data
- `GET /api/pubmed/search` - Search PubMed
- `GET /api/clinicaltrials/search` - Search ClinicalTrials.gov

### Processing
- `GET /api/processing/{job_id}` - Check processing status

---

## 🚀 Deployment

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
LAVA_SECRET_KEY=aks_live_...
LAVA_CONNECTION_SECRET=cons_live_...
LAVA_PRODUCT_SECRET=ps_live_...
ENABLE_LAVA=true
EOF

# Run server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Production Considerations

**Backend:**
- Use PostgreSQL instead of SQLite
- Add Redis for caching
- Deploy with Gunicorn + Nginx
- Set up CORS properly
- Use environment-specific configs

**Frontend:**
- Build with `npm run build`
- Serve with Nginx or CDN
- Enable gzip compression
- Add error tracking (Sentry)

**RAG:**
- Consider vector database (Pinecone, Weaviate) for large scale
- Implement chunking strategies per document type
- Add embedding caching

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### LLM Connection Test
```bash
cd backend
source venv/bin/activate
python -c "from app.services import LavaService; import asyncio; asyncio.run(LavaService().test_connection())"
```

---

## 📊 Performance Optimization

### Current Optimizations
- ✅ Background PDF processing (FastAPI BackgroundTasks)
- ✅ Lazy RAG index loading
- ✅ Graph merging on-demand
- ✅ Frontend state caching (Zustand)

### Future Improvements
- [ ] Implement pagination for large graphs
- [ ] Add graph node clustering for visualization
- [ ] Cache LLM responses
- [ ] Implement incremental RAG updates
- [ ] Add WebSocket for real-time processing updates

---

## 🔒 Security

### Current Measures
- ✅ OAuth authentication (Google)
- ✅ User-scoped data access
- ✅ SQL injection prevention (SQLAlchemy ORM)
- ✅ CORS configuration
- ✅ API key security (environment variables)

### Recommendations
- [ ] Add rate limiting
- [ ] Implement API key rotation
- [ ] Add request validation middleware
- [ ] Enable HTTPS in production
- [ ] Add audit logging

---

## 📚 Key Technologies

**Backend:**
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- NetworkX (graph analysis)
- spaCy (NER)
- sentence-transformers (embeddings)
- PyMuPDF (PDF processing)
- httpx (async HTTP)

**Frontend:**
- React 18 (UI framework)
- TypeScript (type safety)
- Vite (build tool)
- Zustand (state management)
- react-force-graph (visualization)
- Tailwind CSS (styling)

**AI/ML:**
- Anthropic Claude (LLM)
- Lava Payments (usage billing)
- sentence-transformers (RAG embeddings)

---

## 🤝 Contributing

### Code Style
- **Backend**: Follow PEP 8, use type hints
- **Frontend**: Use TypeScript strict mode, follow React best practices

### Git Workflow
1. Create feature branch from `main`
2. Make changes with clear commit messages
3. Test thoroughly
4. Submit pull request

---



## 📝 License

[Add license information]

---

## 🆘 Support

For issues or questions:
- Check `README.md` for setup instructions
- Review `RAG_SYSTEM.md` for RAG details
- See `LAVA_SETUP.md` for LLM configuration

---

**Last Updated:** October 2025
**Version:** 2.0 (RAG-Enhanced)

