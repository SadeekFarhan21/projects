# RAG (Retrieval-Augmented Generation) System

## Overview

The RAG system enhances LLM responses by combining:
1. **Vector Embeddings**: Semantic search through PDF content
2. **Knowledge Graph**: Entity relationships and connections
3. **Hybrid Retrieval**: Entity-based + semantic + graph-enhanced search
4. **Context Assembly**: Rich prompts with relevant chunks + relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG System Pipeline                      │
└─────────────────────────────────────────────────────────────┘

1. INDEXING PHASE (During PDF Processing)
   ┌──────────────┐
   │  PDF Upload  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Text Extract │ (PDFProcessor)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Chunking   │ (DocumentChunker)
   │  - Semantic  │
   │  - Overlap   │
   │  - Entities  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │     NER      │ (NERService)
   │  - Extract   │
   │  - Classify  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  RAG Index   │ (RAGService)
   │  - Store     │
   │  - Link      │
   │  - Embed     │
   └──────────────┘

2. RETRIEVAL PHASE (During Query)
   ┌──────────────┐
   │  User Query  │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Extract    │
   │   Entities   │
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────┐
   │      Hybrid Retrieval            │
   │  ┌────────────────────────────┐  │
   │  │  1. Entity-Based Search    │  │
   │  │     - Find chunks with     │  │
   │  │       mentioned entities   │  │
   │  └────────────────────────────┘  │
   │  ┌────────────────────────────┐  │
   │  │  2. Graph Expansion        │  │
   │  │     - Get neighbors        │  │
   │  │     - Find relationships   │  │
   │  └────────────────────────────┘  │
   │  ┌────────────────────────────┐  │
   │  │  3. Semantic Search        │  │
   │  │     - Vector similarity    │  │
   │  │     - (Future: embeddings) │  │
   │  └────────────────────────────┘  │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │     Context Assembly             │
   │  - Relevant text chunks          │
   │  - Entity relationships          │
   │  - Graph context summary         │
   │  - Metadata (pages, docs)        │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │     Build RAG Prompt             │
   │  - Format context                │
   │  - Add instructions              │
   │  - Include citations             │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │        LLM Generation            │
   │  - Process prompt                │
   │  - Generate response             │
   │  - Include citations             │
   └──────────────────────────────────┘
```

## Key Components

### 1. RAGService (`rag_service.py`)
**Purpose**: Core RAG functionality

**Features**:
- Document indexing with entity linking
- Hybrid retrieval (entity + graph + semantic)
- Context assembly for LLM prompts
- Graph-enhanced search
- Index persistence (save/load)

**Key Methods**:
```python
# Index a document
rag_service.index_document(doc_id, text_chunks, entities)

# Retrieve context for a query
context = rag_service.retrieve_context_for_query(
    query="How does p53 interact with miR-21?",
    entities=["p53", "miR-21"],
    top_k=5
)

# Build RAG prompt
prompt = rag_service.build_rag_prompt(
    query=query,
    context=context,
    task_type="answer"  # or "hypothesis", "summary"
)

# Get entity-specific context
entity_context = rag_service.get_entity_context("p53", depth=2)
```

### 2. DocumentChunker (`document_chunker.py`)
**Purpose**: Intelligent text chunking

**Features**:
- Sentence-aware splitting
- Configurable chunk size and overlap
- Entity tracking per chunk
- Page number preservation
- Metadata retention

**Key Methods**:
```python
chunker = DocumentChunker(chunk_size=500, overlap=100)

# Chunk with entity tracking
chunks = chunker.chunk_with_entities(
    text=pdf_text,
    doc_id=document_id,
    entities=extracted_entities
)
```

### 3. ContentInsightAgent (Enhanced)
**Purpose**: Generate insights using RAG

**Integration**: Uses RAG context to generate better hypotheses

## Integration Points

### During PDF Processing

```python
# In process_pdfs_background()

# 1. Extract text and entities (existing)
text = pdf_processor.extract_text(pdf_path)
entities = ner_service.extract_entities(text)

# 2. NEW: Chunk the document
chunker = DocumentChunker(chunk_size=500, overlap=100)
chunks = chunker.chunk_with_entities(
    text=text,
    doc_id=document.id,
    entities=entities
)

# 3. NEW: Index in RAG system
rag_service.index_document(
    doc_id=document.id,
    text_chunks=chunks,
    entities=entities
)

# 4. Build graph (existing)
graph_builder.build_graph(entities, relationships)

# 5. NEW: Set graph context in RAG
rag_service.set_graph_context(
    graph=graph_builder.graph,
    entity_metadata=entity_metadata
)

# 6. NEW: Save RAG index
rag_service.save_index(f"uploads/{project_id}_rag_index.pkl")
```

### During Hypothesis Generation

```python
# In generate_hypotheses()

# 1. Load RAG index for project
rag_service.load_index(f"uploads/{project_id}_rag_index.pkl")

# 2. Set current graph
rag_service.set_graph_context(nx_graph, entity_metadata)

# 3. Extract entities from graph
top_entities = [node for node, degree in sorted(
    nx_graph.degree(), 
    key=lambda x: x[1], 
    reverse=True
)][:10]

# 4. Retrieve relevant context
context = rag_service.retrieve_context_for_query(
    query="Generate research hypotheses",
    entities=top_entities,
    top_k=10,
    include_graph_context=True
)

# 5. Build comprehensive prompt
prompt = rag_service.build_rag_prompt(
    query="Generate novel research hypotheses",
    context=context,
    task_type="hypothesis"
)

# 6. Use LLM with RAG prompt
insights = await llm_service.generate_insights(prompt)
```

### During Chat/Q&A

```python
# In chat endpoint

# 1. Extract entities from user question
question_entities = ner_service.extract_entities(user_question)

# 2. Retrieve relevant context
context = rag_service.retrieve_context_for_query(
    query=user_question,
    entities=question_entities,
    top_k=5
)

# 3. Build RAG prompt
prompt = rag_service.build_rag_prompt(
    query=user_question,
    context=context,
    task_type="answer"
)

# 4. Get LLM response
response = await llm_service.chat(prompt)
```

## Benefits

### 1. **Better Context**
- Actual PDF content, not just graph structure
- Relevant excerpts with citations
- Page numbers for verification

### 2. **Smarter Retrieval**
- Entity-aware search
- Graph-enhanced expansion
- Hybrid scoring

### 3. **Richer Insights**
- Evidence-based hypotheses
- Specific textual support
- Multi-document synthesis

### 4. **Scalability**
- Efficient chunking
- Index persistence
- Incremental updates

## Future Enhancements

### Phase 1 (Current)
- ✅ Entity-based retrieval
- ✅ Graph-enhanced search
- ✅ Context assembly
- ✅ Chunk management

### Phase 2 (Next)
- 🔄 Vector embeddings (OpenAI/HuggingFace)
- 🔄 Semantic similarity search
- 🔄 Re-ranking algorithms
- 🔄 Cross-document synthesis

### Phase 3 (Future)
- ⏳ Fine-tuned embeddings
- ⏳ Multi-modal retrieval (figures, tables)
- ⏳ Temporal reasoning
- ⏳ Citation networks

## Usage Examples

### Example 1: Answer Question with Citations

```python
# User asks: "What is the relationship between p53 and cancer?"

# System retrieves:
context = {
    "chunks": [
        {
            "text": "The p53 protein is a tumor suppressor that plays a crucial role...",
            "doc_id": "paper1.pdf",
            "page": 3,
            "entities": ["p53", "cancer", "tumor suppressor"]
        }
    ],
    "relationships": [
        {"source": "p53", "target": "apoptosis", "type": "REGULATES"}
    ],
    "graph_context": "p53 is connected to: apoptosis (REGULATES), cell cycle (CONTROLS)..."
}

# LLM generates answer with citations:
"The p53 protein is a tumor suppressor that plays a crucial role in cancer prevention 
[paper1.pdf, page 3]. It regulates apoptosis and controls the cell cycle..."
```

### Example 2: Generate Hypotheses

```python
# System retrieves context about top entities
context = rag_service.retrieve_context_for_query(
    query="Generate hypotheses",
    entities=["p53", "miR-21", "EGFR"],
    top_k=15
)

# LLM generates evidence-based hypotheses:
[
    {
        "title": "miR-21 may mediate EGFR-induced p53 suppression",
        "evidence": ["Excerpt from paper1.pdf showing EGFR upregulates miR-21",
                    "Excerpt from paper2.pdf showing miR-21 downregulates p53"],
        "confidence": 0.85
    }
]
```

## Implementation Checklist

- [ ] Initialize RAG service in main.py
- [ ] Integrate chunking during PDF processing
- [ ] Index documents as they're processed
- [ ] Update hypothesis generation to use RAG
- [ ] Update chat endpoint to use RAG
- [ ] Add RAG index persistence
- [ ] Test with sample PDFs
- [ ] Add vector embeddings (optional, Phase 2)
- [ ] Add API endpoints for RAG queries
- [ ] Update frontend to show citations

## Performance Considerations

1. **Chunking**: ~1-2 seconds per PDF
2. **Indexing**: ~0.5 seconds per document
3. **Retrieval**: ~0.1-0.5 seconds per query
4. **LLM Generation**: 2-10 seconds (depends on LLM)

**Total**: ~3-15 seconds for RAG-enhanced response (vs 2-5s without RAG)

## Storage

- **Chunks**: ~10-50 KB per PDF
- **Index**: ~50-200 KB per project
- **Embeddings** (future): ~1-5 MB per project

## Conclusion

The RAG system transforms the application from a simple graph visualizer into an intelligent research assistant that can:
- Answer questions with evidence
- Generate hypotheses based on actual content
- Provide citations and sources
- Synthesize information across multiple papers
- Leverage both structured (graph) and unstructured (text) data

This makes the insights significantly more useful and trustworthy!

