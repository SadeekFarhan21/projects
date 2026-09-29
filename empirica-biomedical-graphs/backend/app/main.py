from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import os
import shutil
from pathlib import Path
import asyncio
from datetime import datetime
import networkx as nx

def _parse_confidence(confidence_value):
    """
    Parse confidence value from various formats to float.
    Handles both numeric and string confidence values.
    """
    if isinstance(confidence_value, (int, float)):
        return float(confidence_value)
    
    if isinstance(confidence_value, str):
        confidence_lower = confidence_value.lower()
        
        # Map string confidence to numeric values
        if confidence_lower in ['very high', 'very-high', 'veryhigh']:
            return 0.9
        elif confidence_lower in ['high', 'strong']:
            return 0.8
        elif confidence_lower in ['medium-high', 'medium high', 'mediumhigh']:
            return 0.7
        elif confidence_lower in ['medium', 'moderate']:
            return 0.6
        elif confidence_lower in ['medium-low', 'medium low', 'mediumlow']:
            return 0.4
        elif confidence_lower in ['low', 'weak']:
            return 0.3
        elif confidence_lower in ['very low', 'very-low', 'verylow']:
            return 0.2
        else:
            # Default to medium confidence for unknown strings
            return 0.5
    
    # Default fallback
    return 0.5

def _parse_evidence(evidence_value):
    """
    Parse evidence value to ensure it's a list.
    Handles both string and list evidence values.
    """
    if isinstance(evidence_value, list):
        return evidence_value
    elif isinstance(evidence_value, str):
        # Split string evidence into sentences or return as single item
        if '. ' in evidence_value or '; ' in evidence_value:
            # Split by common sentence delimiters
            sentences = evidence_value.replace('; ', '. ').split('. ')
            return [s.strip() for s in sentences if s.strip()]
        else:
            # Return as single sentence
            return [evidence_value]
    else:
        # Default to empty list
        return []

from app.config import settings
from app.models import (
    GraphData,
    GraphAnalytics,
    ProcessingStatus,
    ProjectMetadata,
    PDFMetadata,
    init_db,
    get_db,
    Project,
    Document,
    PDFGraphNode,
    PDFGraphEdge,
    SessionLocal,
)
from app.services.auth_service import get_current_user, get_current_user_optional
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.document_chunker import DocumentChunker
from app.services.ner_service import NERService
from app.services.relationship_extractor import RelationshipExtractor
from app.models.database import User
from app.services import (
    PDFProcessor,
    NERService,
    RelationshipExtractor,
    GraphBuilder,
    LLMService,
    GraphConversationalAgent,
    ContentInsightAgent,
    RAGService,
    DocumentChunker,
    PubMedService,
    ClinicalTrialsService,
    AgenticAIService,
)
from sqlalchemy.orm import Session
from app.models import (
    ChatRequest,
    ChatResponse,
    HypothesesRequest,
    HypothesesResponse,
    NerPreviewRequest,
    NerPreviewResponse,
    ProjectExport,
    ProjectImportRequest,
    PaperDiscoveryRequest,
    PaperDiscoveryResponse,
    DiscoveredPaper,
    TrialDiscoveryRequest,
    TrialDiscoveryResponse,
    ClinicalTrial,
)

# Initialize FastAPI app
app = FastAPI(
    title="Empirica API",
    description="Transform biomedical PDFs into interactive knowledge graphs",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"🔍 Received request: {request.method} {request.url}")
    print(f"📋 Headers: {dict(request.headers)}")
    print(f"🌐 Origin: {request.headers.get('origin', 'No origin')}")
    print(f"🔑 Authorization: {request.headers.get('authorization', 'No auth')[:50]}...")
    
    response = await call_next(request)
    print(f"✅ Response: {response.status_code}")
    return response

# Initialize services
pdf_processor = PDFProcessor()
from app.config import settings as _settings
ner_service = NERService(model_name=_settings.scispacy_model)
relationship_extractor = RelationshipExtractor()
graph_builder = GraphBuilder()
llm_service = LLMService()
rag_service = RAGService(llm_service=llm_service)
document_chunker = DocumentChunker(chunk_size=500, overlap=100)
pubmed_service = PubMedService()
ctgov_service = ClinicalTrialsService()

# Storage for processing jobs
processing_jobs = {}

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    print("✅ Database initialized")
    print(f"✅ Empirica API running on http://{settings.api_host}:{settings.api_port}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Empirica API",
        "version": "1.0.0",
        "llm_enabled": settings.enable_llm_extraction,
    }

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint for testing"""
    return {
        "status": "healthy",
        "message": "API is running",
        "cors_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/test-auth")
async def test_auth(
    current_user: User = Depends(get_current_user)
):
    """Test authentication endpoint"""
    return {
        "status": "authenticated",
        "user_id": current_user.id,
        "user_name": current_user.name,
        "user_email": current_user.email
    }

@app.get("/api/projects-test")
async def list_projects_test(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Test projects endpoint with authentication"""
    try:
        projects = db.query(Project).filter(Project.user_id == current_user.id).all()
        return {
            "status": "success",
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "created_at": p.created_at.isoformat()
                }
                for p in projects
            ],
            "count": len(projects)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "projects": [],
            "count": 0
        }


@app.post("/api/process", response_model=ProcessingStatus)
async def process_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Process uploaded PDF files and generate knowledge graph
    
    This endpoint:
    1. Accepts multiple PDF files
    2. Extracts text and entities
    3. Identifies relationships
    4. Builds an interactive graph
    5. Returns graph data for visualization
    """
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
            )
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files with metadata
    saved_files = []
    for file in files:
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append({
            "path": str(file_path),
            "original_name": file.filename
        })
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Queued for processing"
    )
    
    # Start background processing
    background_tasks.add_task(
        process_pdfs_background,
        job_id,
        saved_files,
        project_name or f"Project_{job_id[:8]}",
        current_user.id
    )
    
    return processing_jobs[job_id]


async def process_pdfs_background(
    job_id: str,
    pdf_files: List[dict],  # Now receives list of {path, original_name}
    project_name: str,
    user_id: str
):
    """Background task to process PDFs - creates individual graph for each PDF"""
    from app.models.database import Project, Document, PDFGraphNode, PDFGraphEdge
    from sqlalchemy.orm import Session
    
    db = SessionLocal()
    try:
        # Create or get project (check by name and user_id to avoid conflicts)
        project = db.query(Project).filter(
            Project.name == project_name,
            Project.user_id == user_id
        ).first()
        if not project:
            project = Project(
                id=str(uuid.uuid4()),
                name=project_name,
                description=f"Project with {len(pdf_files)} PDFs",
                user_id=user_id
            )
            db.add(project)
            db.commit()
        
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 0.0
        processing_jobs[job_id].message = "Processing PDFs individually..."
        
        pdf_graphs = []
        total_pdfs = len(pdf_files)
        
        # Process each PDF individually
        for idx, pdf_file in enumerate(pdf_files):
            pdf_path = pdf_file["path"]
            original_name = pdf_file["original_name"]
            base_progress = idx / total_pdfs
            progress_step = 1.0 / total_pdfs
            
            processing_jobs[job_id].message = f"Processing PDF {idx + 1}/{total_pdfs}: {original_name}"
            processing_jobs[job_id].progress = base_progress
            
            # Create document record with original filename
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                project_id=project.id,
                filename=original_name,  # Use original filename for display
                file_path=pdf_path,  # Keep full path for file access
                processed=0,
                selected=1
            )
            db.add(document)
            db.commit()
            
            try:
                # Extract text from this PDF
                pdf_result = pdf_processor.process_pdfs([pdf_path])[0]
                
                # Check for errors or missing data
                if "error" in pdf_result or not pdf_result.get("sentences"):
                    print(f"✗ Error processing {original_name}: {pdf_result.get('error', 'No sentences extracted')}")
                    document.processed = -1
                    db.commit()
                    continue
                
                sentences = pdf_result.get("sentences", [])
                full_text = " ".join(sentences)
                
                # Extract entities for this PDF
                sentence_entities = ner_service.extract_entities_from_sentences(sentences)
                filtered_entities = ner_service.filter_entities(sentence_entities)
                unique_entities = ner_service.get_unique_entities(filtered_entities)
                
                # RAG: Chunk the document with entity tracking
                # Flatten filtered_entities to get all entities from all sentences
                entity_list = []
                for sent_data in filtered_entities:
                    for ent in sent_data["entities"]:
                        entity_list.append({
                            "text": ent["text"],
                            "start": ent.get("start", 0),
                            "end": ent.get("end", 0),
                            "type": ent.get("type", "ENTITY")
                        })
                chunks = document_chunker.chunk_with_entities(
                    text=full_text,
                    doc_id=doc_id,
                    entities=entity_list
                )
                
                # RAG: Index document chunks
                rag_service.index_document(
                    doc_id=doc_id,
                    text_chunks=chunks,
                    entities=entity_list
                )
                
                print(f"  → Indexed {len(chunks)} chunks for RAG")
                
                # Extract relationships for this PDF
                relationships = relationship_extractor.extract_all_relationships(filtered_entities)
                
                # Build graph for this PDF
                pdf_graph = graph_builder.build_graph(unique_entities, relationships)
                
                # Save nodes to database
                for node in pdf_graph.nodes:
                    pdf_node = PDFGraphNode(
                        document_id=doc_id,
                        entity_id=node.id,
                        entity_type=node.group.value,
                        count=node.metadata.get("count", 1),
                        degree=node.value
                    )
                    db.add(pdf_node)
                
                # Save edges to database
                for edge in pdf_graph.edges:
                    pdf_edge = PDFGraphEdge(
                        document_id=doc_id,
                        source_id=edge.source,
                        target_id=edge.target,
                        weight=edge.value,
                        evidence=edge.metadata.get("all_evidence", []),
                        relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                    )
                    db.add(pdf_edge)
                
                document.processed = 1
                db.commit()
                
                pdf_graphs.append(pdf_graph)
                
                print(f"✓ Processed {original_name}: {len(pdf_graph.nodes)} nodes, {len(pdf_graph.edges)} edges")
                
            except Exception as e:
                print(f"✗ Error processing {original_name}: {e}")
                import traceback
                traceback.print_exc()
                document.processed = -1
                db.commit()
        
        # Merge all PDF graphs for the initial result
        if pdf_graphs:
            processing_jobs[job_id].message = "Merging graphs..."
            processing_jobs[job_id].progress = 0.95
            
            merged_graph = pdf_graphs[0]
            for pdf_graph in pdf_graphs[1:]:
                # Extract entities and relationships for merging
                base_entities = {n.id: {"original_name": n.id, "type": n.group.value, "count": n.metadata.get("count", 1)} for n in merged_graph.nodes}
                base_rels = [{"source": e.source, "target": e.target, "weight": e.value, "evidence": e.metadata.get("all_evidence", []), "relationship_type": e.metadata.get("relationship_type", "CO_OCCURRENCE")} for e in merged_graph.edges]
                
                new_entities = {n.id: {"original_name": n.id, "type": n.group.value, "count": n.metadata.get("count", 1)} for n in pdf_graph.nodes}
                new_rels = [{"source": e.source, "target": e.target, "weight": e.value, "evidence": e.metadata.get("all_evidence", []), "relationship_type": e.metadata.get("relationship_type", "CO_OCCURRENCE")} for e in pdf_graph.edges]
                
                merged_graph = graph_builder.merge_graphs(base_entities, base_rels, new_entities, new_rels)
            
            processing_jobs[job_id].result = merged_graph
            
            # RAG: Set graph context and save index
            entity_metadata = {n.id: {"type": n.group.value, "count": n.metadata.get("count", 1)} 
                             for n in merged_graph.nodes}
            rag_service.set_graph_context(graph_builder.graph, entity_metadata)
            
            # Save RAG index for this project
            rag_index_path = f"uploads/{project.id}_rag_index.pkl"
            rag_service.save_index(rag_index_path)
            print(f"✓ Saved RAG index to {rag_index_path}")
        
        # Complete
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        processing_jobs[job_id].message = f"Processed {total_pdfs} PDFs successfully!"
        
        # Add project_id to result metadata if result exists
        if processing_jobs[job_id].result:
            processing_jobs[job_id].result.metadata["project_id"] = project.id
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Error: {str(e)}"
        print(f"Processing error for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        # Don't cleanup uploaded files anymore since we keep them
        # for path in pdf_paths:
        #     try:
        #         os.remove(path)
        #     except:
        #         pass


@app.get("/api/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.post("/api/graph/filter")
async def filter_graph(
    graph_data: GraphData,
    min_degree: int = 1,
    entity_types: Optional[List[str]] = None,
    top_n: Optional[int] = None,
):
    """Filter an existing graph based on criteria"""
    try:
        # Rebuild internal graph from data
        entities = {
            node.id: {
                "original_name": node.id,
                "type": node.group.value,
                "count": node.metadata.get("count", 1)
            }
            for node in graph_data.nodes
        }
        
        relationships = [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.value,
                "evidence": edge.metadata.get("all_evidence", [edge.title]),
                "relationship_type": edge.metadata.get("relationship_type", "CO_OCCURRENCE")
            }
            for edge in graph_data.edges
        ]
        
        graph_builder.build_graph(entities, relationships)
        filtered_graph = graph_builder.filter_graph(min_degree, entity_types, top_n)
        
        return filtered_graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics", response_model=GraphAnalytics)
async def compute_graph_analytics(graph_data: GraphData):
    """Compute analytics for a graph"""
    try:
        # Rebuild internal graph
        entities = {
            node.id: {
                "original_name": node.id,
                "type": node.group.value,
                "count": node.metadata.get("count", 1)
            }
            for node in graph_data.nodes
        }
        
        relationships = [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.value,
                "evidence": edge.metadata.get("all_evidence", [edge.title]),
                "relationship_type": edge.metadata.get("relationship_type", "CO_OCCURRENCE")
            }
            for edge in graph_data.edges
        ]
        
        graph_builder.build_graph(entities, relationships)
        analytics = graph_builder.compute_analytics()
        
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects", response_model=List[ProjectMetadata])
async def list_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all saved projects with PDF metadata"""
    from app.models.database import Document, PDFGraphNode, PDFGraphEdge
    from collections import Counter
    
    projects = db.query(Project).filter(Project.user_id == current_user.id).all()
    
    result = []
    for p in projects:
        pdfs = []
        for doc in p.documents:
            # Count entities by type for this PDF
            entity_counts = Counter()
            for node in doc.pdf_nodes:
                entity_counts[node.entity_type] += 1
            
            pdfs.append(PDFMetadata(
                document_id=doc.id,
                filename=doc.filename,
                uploaded_at=doc.uploaded_at.isoformat(),
                processed=doc.processed == 1,
                selected=doc.selected == 1,
                node_count=len(doc.pdf_nodes),
                edge_count=len(doc.pdf_edges),
                entity_counts=dict(entity_counts)
            ))
        
        result.append(ProjectMetadata(
            project_id=p.id,
            name=p.name,
            description=p.description,
            created_at=p.created_at.isoformat(),
            updated_at=p.updated_at.isoformat(),
            pdf_count=len(p.documents),
            pdfs=pdfs
        ))
    
    return result


@app.get("/api/projects/{project_id}/pdfs", response_model=List[PDFMetadata])
async def get_project_pdfs(
    project_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all PDFs for a specific project"""
    from app.models.database import Document
    from collections import Counter
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    pdfs = []
    for doc in project.documents:
        entity_counts = Counter()
        for node in doc.pdf_nodes:
            entity_counts[node.entity_type] += 1
        
        pdfs.append(PDFMetadata(
            document_id=doc.id,
            filename=doc.filename,
            uploaded_at=doc.uploaded_at.isoformat(),
            processed=doc.processed == 1,
            selected=doc.selected == 1,
            node_count=len(doc.pdf_nodes),
            edge_count=len(doc.pdf_edges),
            entity_counts=dict(entity_counts)
        ))
    
    return pdfs


@app.post("/api/projects/{project_id}/select-pdfs")
async def update_pdf_selection(
    project_id: str,
    selected_document_ids: List[str],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update which PDFs are selected for graph visualization"""
    from app.models.database import Document
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update all documents - set selected=0 for all, then selected=1 for selected ones
    for doc in project.documents:
        doc.selected = 1 if doc.id in selected_document_ids else 0
    
    db.commit()
    
    return {"status": "success", "selected_count": len(selected_document_ids)}


@app.post("/api/projects/{project_id}/pdfs")
async def add_pdfs_to_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add more PDFs to an existing project"""
    from app.models.database import Document, PDFGraphNode, PDFGraphEdge
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are supported."
            )
    
    # Create job ID for tracking
    job_id = str(uuid.uuid4())
    
    # Save uploaded files with metadata
    saved_files = []
    for file in files:
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append({
            "path": str(file_path),
            "original_name": file.filename
        })
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Queued for processing"
    )
    
    # Start background processing for these new PDFs
    background_tasks.add_task(
        add_pdfs_to_project_background,
        job_id,
        project_id,
        saved_files,
        current_user.id
    )
    
    return processing_jobs[job_id]


async def add_pdfs_to_project_background(
    job_id: str,
    project_id: str,
    pdf_files: List[dict],  # Now receives list of {path, original_name}
    user_id: str
):
    """Background task to add PDFs to existing project"""
    from app.models.database import Project, Document, PDFGraphNode, PDFGraphEdge
    
    db = SessionLocal()
    try:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == user_id
        ).first()
        if not project:
            processing_jobs[job_id].status = "failed"
            processing_jobs[job_id].message = "Project not found"
            return
        
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 0.0
        processing_jobs[job_id].message = "Processing new PDFs..."
        
        total_pdfs = len(pdf_files)
        
        # Process each PDF individually
        for idx, pdf_file in enumerate(pdf_files):
            pdf_path = pdf_file["path"]
            original_name = pdf_file["original_name"]
            base_progress = idx / total_pdfs
            
            processing_jobs[job_id].message = f"Processing PDF {idx + 1}/{total_pdfs}: {original_name}"
            processing_jobs[job_id].progress = base_progress
            
            # Create document record with original filename
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                project_id=project.id,
                filename=original_name,  # Use original filename for display
                file_path=pdf_path,  # Keep full path for file access
                processed=0,
                selected=1
            )
            db.add(document)
            db.commit()
            
            try:
                # Extract text from this PDF
                pdf_result = pdf_processor.process_pdfs([pdf_path])[0]
                
                # Check for errors or missing data
                if "error" in pdf_result or not pdf_result.get("sentences"):
                    print(f"✗ Error processing {original_name}: {pdf_result.get('error', 'No sentences extracted')}")
                    document.processed = -1
                    db.commit()
                    continue
                
                sentences = pdf_result.get("sentences", [])
                full_text = " ".join(sentences)
                
                # Extract entities for this PDF
                sentence_entities = ner_service.extract_entities_from_sentences(sentences)
                filtered_entities = ner_service.filter_entities(sentence_entities)
                unique_entities = ner_service.get_unique_entities(filtered_entities)
                
                # RAG: Load existing index, chunk and index new document
                rag_index_path = f"uploads/{project_id}_rag_index.pkl"
                rag_service.load_index(rag_index_path)
                
                entity_list = [{"text": ent["text"], "start": 0, "end": 0, "type": ent["label"]} 
                              for ent in filtered_entities]
                chunks = document_chunker.chunk_with_entities(
                    text=full_text,
                    doc_id=doc_id,
                    entities=entity_list
                )
                
                rag_service.index_document(
                    doc_id=doc_id,
                    text_chunks=chunks,
                    entities=entity_list
                )
                
                print(f"  → Indexed {len(chunks)} chunks for RAG")
                
                # Extract relationships for this PDF
                relationships = relationship_extractor.extract_all_relationships(filtered_entities)
                
                # Build graph for this PDF
                pdf_graph = graph_builder.build_graph(unique_entities, relationships)
                
                # Save nodes to database
                for node in pdf_graph.nodes:
                    pdf_node = PDFGraphNode(
                        document_id=doc_id,
                        entity_id=node.id,
                        entity_type=node.group.value,
                        count=node.metadata.get("count", 1),
                        degree=node.value
                    )
                    db.add(pdf_node)
                
                # Save edges to database
                for edge in pdf_graph.edges:
                    pdf_edge = PDFGraphEdge(
                        document_id=doc_id,
                        source_id=edge.source,
                        target_id=edge.target,
                        weight=edge.value,
                        evidence=edge.metadata.get("all_evidence", []),
                        relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                    )
                    db.add(pdf_edge)
                
                document.processed = 1
                db.commit()
                
                print(f"✓ Processed {original_name}: {len(pdf_graph.nodes)} nodes, {len(pdf_graph.edges)} edges")
                
            except Exception as e:
                print(f"✗ Error processing {original_name}: {e}")
                import traceback
                traceback.print_exc()
                document.processed = -1
                db.commit()
        
        # RAG: Save updated index
        rag_index_path = f"uploads/{project_id}_rag_index.pkl"
        rag_service.save_index(rag_index_path)
        print(f"✓ Updated RAG index at {rag_index_path}")
        
        # Complete
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        processing_jobs[job_id].message = f"Added {total_pdfs} PDFs successfully!"
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Error: {str(e)}"
        print(f"Processing error for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


@app.delete("/api/projects/{project_id}/pdfs/{document_id}")
async def delete_pdf_from_project(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a PDF and its graph data from a project"""
    from app.models.database import Document
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.project_id == project_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete the document (cascade will delete nodes and edges)
    filename = document.filename
    db.delete(document)
    db.commit()
    
    return {"status": "success", "message": f"Deleted {filename}"}


@app.get("/api/projects/{project_id}/graph", response_model=GraphData)
async def get_project_graph(
    project_id: str,
    selected_only: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the merged graph from selected PDFs (or all PDFs if selected_only=False)"""
    from app.models.database import Document, PDFGraphNode, PDFGraphEdge
    
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get documents to include
    documents = [doc for doc in project.documents if doc.processed == 1]
    if selected_only:
        documents = [doc for doc in documents if doc.selected == 1]
    
    if not documents:
        return GraphData(nodes=[], edges=[], metadata={"project_id": project_id, "message": "No PDFs selected or processed"})
    
    # Build individual graphs for each PDF
    pdf_graphs = []
    for doc in documents:
        # Reconstruct graph from database
        nodes = []
        for node in doc.pdf_nodes:
            from app.models.schemas import Node, EntityType
            nodes.append(Node(
                id=node.entity_id,
                group=EntityType(node.entity_type),
                value=node.degree,
                metadata={"count": node.count, "degree": node.degree, "source_pdf": doc.filename}
            ))
        
        edges = []
        for edge in doc.pdf_edges:
            from app.models.schemas import Edge
            edges.append(Edge(
                source=edge.source_id,
                target=edge.target_id,
                value=edge.weight,
                title=edge.evidence[0] if edge.evidence else "",
                metadata={
                    "all_evidence": edge.evidence,
                    "relationship_type": edge.relationship_type,
                    "source_pdf": doc.filename
                }
            ))
        
        pdf_graphs.append(GraphData(nodes=nodes, edges=edges, metadata={"source": doc.filename}))
    
    # Merge all graphs
    if len(pdf_graphs) == 1:
        merged = pdf_graphs[0]
    else:
        merged = pdf_graphs[0]
        for pdf_graph in pdf_graphs[1:]:
            # Convert to dict format for merging
            base_entities = {n.id: {"original_name": n.id, "type": n.group.value, "count": n.metadata.get("count", 1)} for n in merged.nodes}
            base_rels = [{"source": e.source, "target": e.target, "weight": e.value, "evidence": e.metadata.get("all_evidence", []), "relationship_type": e.metadata.get("relationship_type", "CO_OCCURRENCE")} for e in merged.edges]
            
            new_entities = {n.id: {"original_name": n.id, "type": n.group.value, "count": n.metadata.get("count", 1)} for n in pdf_graph.nodes}
            new_rels = [{"source": e.source, "target": e.target, "weight": e.value, "evidence": e.metadata.get("all_evidence", []), "relationship_type": e.metadata.get("relationship_type", "CO_OCCURRENCE")} for e in pdf_graph.edges]
            
            merged = graph_builder.merge_graphs(base_entities, base_rels, new_entities, new_rels)
    
    merged.metadata["project_id"] = project_id
    merged.metadata["pdf_count"] = len(documents)
    merged.metadata["selected_pdfs"] = [doc.filename for doc in documents]
    
    return merged


# ==== Conversational Agent Endpoints ====

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_graph(
    payload: dict,
    current_user: User = Depends(get_current_user)
):
    try:
        # Accept flexible JSON payload to avoid validation errors from clients
        message = (payload or {}).get("message", "")
        graph = (payload or {}).get("graph", {})
        conversation_history = (payload or {}).get("conversation_history", [])
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        # Rebuild graph
        entities = {
            n.get("id"): {
                "original_name": n.get("id"),
                "type": (n.get("group") or "UNKNOWN"),
                "count": (n.get("metadata") or {}).get("count", 1),
            }
            for n in nodes
            if n.get("id")
        }
        
        relationships = []
        for e in edges:
            # Handle both string IDs and object references
            source_id = e.get("source")
            target_id = e.get("target")
            
            # If source/target are objects, extract the 'id' field
            if isinstance(source_id, dict):
                source_id = source_id.get("id")
            if isinstance(target_id, dict):
                target_id = target_id.get("id")
            
            relationships.append({
                "source": source_id,
                "target": target_id,
                "weight": e.get("value", 1.0),
                "evidence": (e.get("metadata") or {}).get("all_evidence", [e.get("title", "")]),
                "relationship_type": (e.get("metadata") or {}).get("relationship_type", "CO_OCCURRENCE"),
            })
        
        graph_builder.build_graph(entities, relationships)
        
        # RAG: Try to get project_id and load RAG context
        project_id = (payload or {}).get("project_id")
        rag_context = None
        documents = []
        if project_id:
            try:
                # Get documents for this project
                db = SessionLocal()
                try:
                    documents = db.query(Document).filter(Document.project_id == project_id).all()
                    documents = [
                        {
                            "id": doc.id,
                            "name": doc.filename,
                            "selected": doc.selected
                        }
                        for doc in documents
                    ]
                finally:
                    db.close()
                
                rag_index_path = f"uploads/{project_id}_rag_index.pkl"
                if rag_service.load_index(rag_index_path):
                    # Set graph context
                    rag_service.set_graph_context(graph_builder.graph, entities)
                    
                    # Extract entities from user message (more robust matching)
                    user_entities = []
                    message_lower = message.lower()
                    for entity_name in entities.keys():
                        # Check for exact match, partial match, or word boundary match
                        if (entity_name.lower() in message_lower or 
                            any(word in message_lower for word in entity_name.lower().split()) or
                            any(word in entity_name.lower() for word in message_lower.split())):
                            user_entities.append(entity_name)
                    
                    # Always retrieve context, even for general questions
                    print(f"🔍 User message: '{message}'")
                    print(f"🔍 Found {len(user_entities)} matching entities: {user_entities[:5]}")
                    
                    # Check if user is asking about a specific paper
                    target_doc_id = None
                    message_lower = message.lower()
                    if any(keyword in message_lower for keyword in ['paper1', 'paper 1', 'first paper', 'document1', 'document 1']):
                        # Find paper1 document ID
                        for doc in documents:
                            if 'paper1' in doc.get('name', '').lower() or 'paper 1' in doc.get('name', '').lower():
                                target_doc_id = doc.get('id')
                                print(f"🎯 Targeting specific paper: {doc.get('name')} (ID: {target_doc_id})")
                                break
                    elif any(keyword in message_lower for keyword in ['paper2', 'paper 2', 'second paper', 'document2', 'document 2']):
                        # Find paper2 document ID
                        for doc in documents:
                            if 'paper2' in doc.get('name', '').lower() or 'paper 2' in doc.get('name', '').lower():
                                target_doc_id = doc.get('id')
                                print(f"🎯 Targeting specific paper: {doc.get('name')} (ID: {target_doc_id})")
                                break
                    
                    # Use more entities for better context, or all if few entities
                    context_entities = user_entities if user_entities else list(entities.keys())[:10]
                    
                    rag_context = rag_service.retrieve_context_for_query(
                        query=message,
                        entities=context_entities,
                        top_k=8,  # Increased for better context
                        include_graph_context=True,
                        target_doc_id=target_doc_id
                    )
                    print(f"✓ Retrieved {len(rag_context.get('chunks', []))} chunks for chat")
            except Exception as e:
                print(f"RAG retrieval for chat failed: {e}")

        # Create agent with LLM service for intelligent chat
        agent = GraphConversationalAgent(graph_builder.graph, llm_service=llm_service)
        
        # If RAG context available and LLM enabled, use RAG-enhanced prompt
        if rag_context and llm_service.enabled:
            try:
                # Build RAG prompt for answering
                rag_prompt = rag_service.build_rag_prompt(
                    query=message,
                    context=rag_context,
                    task_type="answer"
                )
                
                print(f"🤖 Using RAG-enhanced prompt with {len(rag_context.get('chunks', []))} chunks")
                
                # Use the RAG-enhanced prompt directly with LLM
                llm_response = await llm_service.chat([{"role": "user", "content": rag_prompt}])
                
                # Create result in expected format
                result = {
                    "answer": llm_response,
                    "relevant_nodes": list(entities.keys())[:5],  # Top entities as relevant nodes
                    "citations": []
                }
                
                # Enhance answer with RAG citations if available
                if rag_context.get("chunks"):
                    citations = [f"{chunk.get('doc_id', 'unknown')} (page {chunk.get('page', '?')})" 
                               for chunk in rag_context["chunks"][:3]]
                    result["citations"] = citations
            except Exception as e:
                print(f"RAG-enhanced chat failed: {e}")
                # Fallback to regular chat
                result = await agent.chat(message, conversation_history)
        else:
            # Use LLM-powered chat if available, otherwise fallback to pattern matching
            result = await agent.chat(message, conversation_history)
        
        return ChatResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            relevant_nodes=result.get("relevant_nodes", []),
            relevant_edges=result.get("relevant_edges", []),
            tool_calls=result.get("tool_calls", []),
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hypotheses", response_model=HypothesesResponse)
async def generate_hypotheses(
    payload: dict,
    current_user: User = Depends(get_current_user)
):
    try:
        graph = (payload or {}).get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        focus_entity = (payload or {}).get("focus_entity")
        max_results = int((payload or {}).get("max_results", 10))
        project_id = (payload or {}).get("project_id")

        # Build NetworkX graph for analysis
        nx_graph = nx.Graph()
        
        print(f"DEBUG: Processing {len(nodes)} nodes and {len(edges)} edges")
        if nodes:
            print(f"DEBUG: Sample node: {nodes[0]}")
        if edges:
            print(f"DEBUG: Sample edge: {edges[0]}")
        
        # Add nodes (just the IDs, metadata will be handled separately)
        for node in nodes:
            node_id = node.get("id")
            # Handle case where node might be a dictionary
            if isinstance(node_id, dict):
                node_id = node_id.get("id", str(node_id))
            if node_id:
                nx_graph.add_node(node_id)
        
        # Add edges with metadata
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            
            # Handle case where source/target might be dictionaries
            if isinstance(source, dict):
                source = source.get("id", str(source))
            if isinstance(target, dict):
                target = target.get("id", str(target))
            
            if source and target:
                evidence = edge.get("metadata", {}).get("all_evidence", [])
                # Convert evidence list to string to avoid unhashable type issues
                evidence_str = "; ".join(evidence) if isinstance(evidence, list) else str(evidence)
                nx_graph.add_edge(source, target,
                                 weight=edge.get("value", 1.0),
                                 relationship_type=edge.get("metadata", {}).get("relationship_type", "CO_OCCURRENCE"),
                                 evidence=evidence_str)

        # Get document data if project_id is provided
        documents_data = []
        if project_id:
            db = SessionLocal()
            try:
                documents = db.query(Document).filter(Document.project_id == project_id).all()
                documents_data = [
                    {
                        "name": doc.filename,
                        "type": "PDF",
                        "id": doc.id,
                        "selected": doc.selected
                    }
                    for doc in documents
                ]
            finally:
                db.close()

        # RAG: Load index and retrieve relevant context
        rag_context = None
        if project_id:
            try:
                rag_index_path = f"uploads/{project_id}_rag_index.pkl"
                if rag_service.load_index(rag_index_path):
                    print(f"✓ Loaded RAG index from {rag_index_path}")
                    
                    # Set graph context
                    entity_metadata = {node.get("id"): {"type": node.get("group", "UNKNOWN"), 
                                                        "count": node.get("metadata", {}).get("count", 1)} 
                                     for node in nodes if node.get("id")}
                    rag_service.set_graph_context(nx_graph, entity_metadata)
                    
                    # Get top entities from graph
                    top_entities = [node_id for node_id, degree in sorted(
                        nx_graph.degree(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )][:10]
                    
                    # Retrieve relevant context
                    rag_context = rag_service.retrieve_context_for_query(
                        query="Generate research hypotheses and insights",
                        entities=top_entities if not focus_entity else [focus_entity] + top_entities,
                        top_k=10,
                        include_graph_context=True
                    )
                    
                    print(f"✓ Retrieved {len(rag_context.get('chunks', []))} relevant chunks from RAG")
            except Exception as e:
                print(f"RAG retrieval failed: {e}")
        
        # Use ContentInsightAgent for better insights
        # Pass the original node data for metadata
        print(f"DEBUG: Creating ContentInsightAgent with {len(nx_graph.nodes())} nodes, {len(nx_graph.edges())} edges")
        print(f"DEBUG: Documents data: {len(documents_data)} documents")
        print(f"DEBUG: Original nodes: {len(nodes)} nodes")
        
        insight_agent = ContentInsightAgent(nx_graph, documents_data, nodes)
        print("DEBUG: ContentInsightAgent created successfully")
        
        # Try to use LLM if available with RAG context, otherwise fall back to pattern-based insights
        if llm_service.enabled:
            try:
                # Use RAG prompt if context available, otherwise use basic prompt
                if rag_context:
                    prompt = rag_service.build_rag_prompt(
                        query=f"Generate research hypotheses{' focusing on ' + focus_entity if focus_entity else ''}",
                        context=rag_context,
                        task_type="hypothesis"
                    )
                    print(f"✓ Using RAG-enhanced prompt with {len(rag_context.get('chunks', []))} chunks")
                else:
                    # Fallback to ContentInsightAgent prompt
                    prompt = insight_agent.get_llm_prompt(focus_entity)
                
                # Use LLM to generate insights
                llm_response = await llm_service.generate_insights(prompt)
                
                if llm_response:
                    insights = llm_response
                else:
                    # Fallback to pattern-based insights
                    insights = insight_agent.generate_insights(focus_entity, max_results)
            except Exception as e:
                print(f"LLM insight generation failed: {e}")
                # Fallback to pattern-based insights
                insights = insight_agent.generate_insights(focus_entity, max_results)
        else:
            # Use pattern-based insights
            insights = insight_agent.generate_insights(focus_entity, max_results)

        # Normalize insights to match expected format
        normalized = []
        for insight in insights[:max_results]:
            normalized.append({
                "title": insight.get("title", "Research Insight"),
                "explanation": insight.get("description", insight.get("explanation", "")),
                "entities": insight.get("entities", []),
                "evidence_sentences": _parse_evidence(insight.get("evidence", [])),
                "edge_pairs": [],  # Not used in new format
                "confidence": _parse_confidence(insight.get("confidence", 0.5)),
                "type": insight.get("type", "insight")
            })

        return HypothesesResponse(hypotheses=normalized)
    except Exception as e:
        import traceback
        print(f"Hypothesis generation error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ner/preview", response_model=NerPreviewResponse)
async def ner_preview(req: NerPreviewRequest):
    try:
        doc = ner_service.nlp.make_doc(req.text)
        if not ner_service.nlp.has_pipe("sentencizer"):
            try:
                ner_service.nlp.add_pipe("sentencizer")
            except Exception:
                pass
        doc = ner_service.nlp(req.text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        sent_ents = ner_service.extract_entities_from_sentences(sentences)

        original = ner_service.min_entity_occurrences
        ner_service.min_entity_occurrences = max(1, req.min_occurrences)
        filtered = ner_service.filter_entities(sent_ents)
        unique = ner_service.get_unique_entities(filtered)
        ner_service.min_entity_occurrences = original

        def to_schema(items: List[dict]) -> List[dict]:
            return [
                {
                    "sentence_id": i["sentence_id"],
                    "sentence": i["sentence"],
                    "entities": i["entities"],
                }
                for i in items
            ]

        # Build debug info
        from collections import Counter
        label_counter = Counter()
        for s in sentences[: min(50, len(sentences))]:
            d = ner_service.nlp(s)
            for ent in d.ents:
                label_counter[ent.label_] += 1

        sample_by_label = {}
        for s in sentences[: min(50, len(sentences))]:
            d = ner_service.nlp(s)
            for ent in d.ents:
                lbl = ent.label_
                sample_by_label.setdefault(lbl, [])
                if len(sample_by_label[lbl]) < 5:
                    sample_by_label[lbl].append(ent.text)

        return NerPreviewResponse(
            sentences=to_schema(filtered),
            unique_entities=unique,
            raw_sentences=to_schema(sent_ents) if req.return_raw else None,
            debug={
                "label_counts": dict(label_counter),
                "samples": sample_by_label,
                "model": _settings.scispacy_model,
                "min_entity_occurrences": original,
                "used_min_occurrences": max(1, req.min_occurrences),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==== Project Export/Import Endpoints ====

@app.delete("/api/projects/{project_id}")
async def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a project and all its associated data"""
    from app.models.database import Document, PDFGraphNode, PDFGraphEdge
    import os
    
    # Find project and verify ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # Delete associated files
        for document in project.documents:
            if document.file_path and os.path.exists(document.file_path):
                os.remove(document.file_path)
        
        # Delete all associated data (cascade should handle this, but being explicit)
        db.query(PDFGraphNode).filter(
            PDFGraphNode.document_id.in_([doc.id for doc in project.documents])
        ).delete(synchronize_session=False)
        
        db.query(PDFGraphEdge).filter(
            PDFGraphEdge.document_id.in_([doc.id for doc in project.documents])
        ).delete(synchronize_session=False)
        
        # Delete documents
        db.query(Document).filter(Document.project_id == project_id).delete()
        
        # Delete project
        db.delete(project)
        db.commit()
        
        return {"status": "success", "message": f"Project '{project.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


@app.put("/api/projects/{project_id}")
async def rename_project(
    project_id: str,
    new_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Rename a project"""
    # Find project and verify ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if not new_name.strip():
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    
    try:
        # Check if another project with the same name exists for this user
        existing_project = db.query(Project).filter(
            Project.name == new_name.strip(),
            Project.user_id == current_user.id,
            Project.id != project_id
        ).first()
        
        if existing_project:
            raise HTTPException(status_code=400, detail="A project with this name already exists")
        
        # Update project name
        project.name = new_name.strip()
        project.updated_at = datetime.utcnow()
        db.commit()
        
        return {"status": "success", "message": f"Project renamed to '{new_name.strip()}' successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to rename project: {str(e)}")


@app.get("/api/projects/{project_id}/export")
async def export_project(
    project_id: str, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export a project with all individual PDF graphs as JSON"""
    from app.models.schemas import PDFGraphExport, Node, Edge, EntityType
    
    try:
        project = db.query(Project).filter(
            Project.id == project_id,
            Project.user_id == current_user.id
        ).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        pdf_graphs = []
        for doc in project.documents:
            if doc.processed != 1:
                continue
            
            # Build graph for this PDF
            nodes = []
            for node in doc.pdf_nodes:
                nodes.append(Node(
                    id=node.entity_id,
                    group=EntityType(node.entity_type),
                    value=node.degree,
                    metadata={"count": node.count, "degree": node.degree}
                ))
            
            edges = []
            for edge in doc.pdf_edges:
                edges.append(Edge(
                    source=edge.source_id,
                    target=edge.target_id,
                    value=edge.weight,
                    title=edge.evidence[0] if edge.evidence else "",
                    metadata={
                        "all_evidence": edge.evidence,
                        "relationship_type": edge.relationship_type
                    }
                ))
            
            pdf_graph = GraphData(nodes=nodes, edges=edges, metadata={"source": doc.filename})
            
            pdf_graphs.append(PDFGraphExport(
                document_id=doc.id,
                filename=doc.filename,
                uploaded_at=doc.uploaded_at.isoformat(),
                graph=pdf_graph
            ))
        
        # Export RAG index if it exists
        rag_index_data = None
        rag_index_path = f"uploads/{project_id}_rag_index.pkl"
        if os.path.exists(rag_index_path):
            try:
                import pickle
                with open(rag_index_path, 'rb') as f:
                    rag_index_data = pickle.load(f)
                print(f"✓ Exported RAG index with {len(rag_index_data.get('document_chunks', {}))} documents")
            except Exception as e:
                print(f"⚠️  Failed to export RAG index: {e}")
        
        from app.models.schemas import ProjectExport
        return ProjectExport(
            project_name=project.name,
            project_id=project.id,
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
            pdf_graphs=pdf_graphs,
            settings={"rag_index": rag_index_data} if rag_index_data else {}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projects/import")
async def import_project(
    req: dict, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Import a project from JSON with per-PDF graphs"""
    from app.models.database import Project, Document, PDFGraphNode, PDFGraphEdge
    
    try:
        print(f"DEBUG Import: Received keys: {req.keys()}")
        
        project_data = req.get("project_data", req)
        
        # Create new project
        project_id = str(uuid.uuid4())
        project = Project(
            id=project_id,
            name=project_data.get("project_name", f"Imported Project {project_id[:8]}"),
            description="Imported project",
            user_id=current_user.id
        )
        db.add(project)
        db.commit()
        
        # Import each PDF graph
        pdf_graphs = project_data.get("pdf_graphs", [])
        for pdf_graph_data in pdf_graphs:
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                project_id=project.id,
                filename=pdf_graph_data.get("filename", "unknown.pdf"),
                file_path="",  # No actual file for imported projects
                processed=1,
                selected=1
            )
            db.add(document)
            db.commit()
            
            # Import nodes
            graph = pdf_graph_data.get("graph", {})
            for node in graph.get("nodes", []):
                pdf_node = PDFGraphNode(
                    document_id=doc_id,
                    entity_id=node["id"],
                    entity_type=node["group"],
                    count=node.get("metadata", {}).get("count", 1),
                    degree=node.get("value", 1)
                )
                db.add(pdf_node)
            
            # Import edges
            for edge in graph.get("edges", []):
                pdf_edge = PDFGraphEdge(
                    document_id=doc_id,
                    source_id=edge["source"],
                    target_id=edge["target"],
                    weight=edge.get("value", 1.0),
                    evidence=edge.get("metadata", {}).get("all_evidence", []),
                    relationship_type=edge.get("metadata", {}).get("relationship_type", "CO_OCCURRENCE")
                )
                db.add(pdf_edge)
            
            db.commit()
        
        # Import RAG index if it exists in the export
        settings = project_data.get("settings", {})
        rag_index_data = settings.get("rag_index")
        if rag_index_data:
            try:
                import pickle
                rag_index_path = f"uploads/{project.id}_rag_index.pkl"
                with open(rag_index_path, 'wb') as f:
                    pickle.dump(rag_index_data, f)
                print(f"✓ Imported RAG index with {len(rag_index_data.get('document_chunks', {}))} documents")
            except Exception as e:
                print(f"⚠️  Failed to import RAG index: {e}")
        
        return {
            "status": "success",
            "project_id": project.id,
            "project_name": project.name,
            "pdf_count": len(pdf_graphs)
        }
    except Exception as e:
        print(f"ERROR Import: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==== Paper Discovery Endpoints ====

@app.post("/api/discover/papers", response_model=PaperDiscoveryResponse)
async def discover_papers(req: PaperDiscoveryRequest):
    """Discover papers from PubMed and optionally process them"""
    try:
        papers = pubmed_service.discover_and_fetch(req.query, req.max_results)
        
        discovered = [
            DiscoveredPaper(**paper) for paper in papers if paper
        ]
        
        return PaperDiscoveryResponse(
            papers=discovered,
            status="completed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discover/papers/process")
async def process_discovered_papers(
    background_tasks: BackgroundTasks,
    papers: List[DiscoveredPaper],
    merge_with_existing: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Process discovered papers through NER pipeline"""
    try:
        job_id = str(uuid.uuid4())
        
        # Combine all abstracts into sentences
        all_text = "\n\n".join([f"{p.title}. {p.abstract}" for p in papers])
        
        processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Processing discovered papers"
        )
        
        background_tasks.add_task(
            process_text_background,
            job_id,
            all_text,
            f"Discovered_Papers_{job_id[:8]}",
            current_user.id
        )
        
        return processing_jobs[job_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_text_background(job_id: str, text: str, project_name: str, user_id: str):
    """Background task to process text through NER pipeline"""
    try:
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 0.2
        processing_jobs[job_id].message = "Extracting entities..."
        
        # Split into sentences
        doc = ner_service.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        processing_jobs[job_id].progress = 0.4
        sentence_entities = ner_service.extract_entities_from_sentences(sentences)
        
        processing_jobs[job_id].progress = 0.6
        filtered_entities = ner_service.filter_entities(sentence_entities)
        unique_entities = ner_service.get_unique_entities(filtered_entities)
        
        processing_jobs[job_id].progress = 0.8
        processing_jobs[job_id].message = "Extracting relationships..."
        relationships = relationship_extractor.extract_all_relationships(filtered_entities)
        
        processing_jobs[job_id].progress = 0.9
        processing_jobs[job_id].message = "Building graph..."
        graph_data = graph_builder.build_graph(unique_entities, relationships)
        
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 1.0
        processing_jobs[job_id].message = "Complete!"
        processing_jobs[job_id].result = graph_data
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Error: {str(e)}"


# ==== Clinical Trials Discovery Endpoints ====

@app.post("/api/discover/trials", response_model=TrialDiscoveryResponse)
async def discover_trials(req: TrialDiscoveryRequest):
    """Discover clinical trials and convert to graph nodes"""
    try:
        trials = ctgov_service.search_trials(
            req.condition,
            req.max_results,
            req.phases,
            req.status
        )
        
        trial_objects = [ClinicalTrial(**t) for t in trials if t.get("nct_id")]
        
        # Convert trials to graph nodes and edges
        nodes = []
        edges = []
        
        for trial in trial_objects:
            # Add trial node
            trial_node_id = f"TRIAL:{trial.nct_id}"
            nodes.append({
                "id": trial_node_id,
                "group": "ENTITY",
                "value": len(trial.interventions) + 1,
                "metadata": {
                    "type": "clinical_trial",
                    "nct_id": trial.nct_id,
                    "phase": trial.phase,
                    "status": trial.status,
                    "url": trial.url
                }
            })
            
            # Add condition node
            if trial.condition:
                nodes.append({
                    "id": trial.condition,
                    "group": "DISEASE",
                    "value": 1,
                    "metadata": {"type": "disease"}
                })
                edges.append({
                    "source": trial_node_id,
                    "target": trial.condition,
                    "value": 1.0,
                    "title": f"{trial.nct_id} studies {trial.condition}",
                    "metadata": {
                        "relationship_type": "CLINICAL_TRIAL_STUDIES",
                        "all_evidence": [trial.brief_summary[:200]]
                    }
                })
            
            # Add intervention nodes
            for intervention in trial.interventions:
                nodes.append({
                    "id": intervention,
                    "group": "CHEMICAL",
                    "value": 1,
                    "metadata": {"type": "intervention"}
                })
                edges.append({
                    "source": trial_node_id,
                    "target": intervention,
                    "value": 1.0,
                    "title": f"{trial.nct_id} tests {intervention}",
                    "metadata": {
                        "relationship_type": "CLINICAL_TRIAL_TESTS",
                        "all_evidence": [trial.brief_summary[:200]]
                    }
                })
        
        # Deduplicate nodes
        unique_nodes = {n["id"]: n for n in nodes}
        graph = GraphData(
            nodes=list(unique_nodes.values()),
            edges=edges,
            metadata={"source": "clinicaltrials.gov"}
        )
        
        return TrialDiscoveryResponse(
            trials=trial_objects,
            graph=graph
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==== Agentic AI Endpoints ====

# Storage for agentic research jobs
agentic_research_jobs = {}

@app.post("/api/agentic/research")
async def start_agentic_research(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Start autonomous research on a topic"""
    try:
        research_topic = request.get("research_topic", "")
        max_papers = int(request.get("max_papers", 10))
        search_strategy = request.get("search_strategy", "comprehensive")
        project_id = request.get("project_id")
        
        if not research_topic:
            raise HTTPException(status_code=400, detail="Research topic is required")
        
        # Generate unique research ID
        research_id = str(uuid.uuid4())
        
        # Initialize job status
        agentic_research_jobs[research_id] = {
            "status": "starting",
            "current_stage": "Starting research...",
            "progress": {
                "papers_found": 0,
                "papers_analyzed": 0,
                "entities_extracted": 0,
                "relationships_found": 0
            },
            "results": None,
            "error": None,
            "started_at": datetime.now().isoformat()
        }
        
        # Start background research
        background_tasks.add_task(
            process_agentic_research,
            research_id,
            research_topic,
            max_papers,
            search_strategy,
            project_id
        )
        
        return {
            "research_id": research_id,
            "status": "started",
            "message": f"Research started on: {research_topic}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agentic/research/{research_id}/status")
async def get_agentic_research_status(
    research_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of agentic research"""
    try:
        if research_id not in agentic_research_jobs:
            raise HTTPException(status_code=404, detail="Research job not found")
        
        job = agentic_research_jobs[research_id]
        response = {
            "research_id": research_id,
            "status": job["status"],
            "current_stage": job.get("current_stage", "Processing..."),
            "progress": job["progress"],
            "started_at": job["started_at"],
            "error": job.get("error")
        }
        
        # Debug logging
        print(f"📊 Status request for {research_id}: status={response['status']}, papers_found={response['progress']['papers_found']}, papers_analyzed={response['progress']['papers_analyzed']}, entities={response['progress']['entities_extracted']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agentic/research/{research_id}/results")
async def get_agentic_research_results(
    research_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the results of completed agentic research"""
    try:
        if research_id not in agentic_research_jobs:
            raise HTTPException(status_code=404, detail="Research job not found")
        
        job = agentic_research_jobs[research_id]
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Research not completed yet")
        
        if not job["results"]:
            raise HTTPException(status_code=500, detail="No results available")
        
        return job["results"]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agentic/research/{research_id}/expand")
async def expand_agentic_research(
    research_id: str,
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Find related papers to expand existing research"""
    try:
        if research_id not in agentic_research_jobs:
            raise HTTPException(status_code=404, detail="Research job not found")
        
        job = agentic_research_jobs[research_id]
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Research must be completed first")
        
        max_new_papers = int(request.get("max_new_papers", 5))
        
        # Start background expansion
        background_tasks.add_task(
            expand_agentic_research,
            research_id,
            max_new_papers
        )
        
        return {
            "research_id": research_id,
            "status": "expanding",
            "message": f"Finding {max_new_papers} related papers"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agentic/research/{research_id}/save")
async def save_agentic_research(
    research_id: str,
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save agentic research results as a new project"""
    try:
        if research_id not in agentic_research_jobs:
            raise HTTPException(status_code=404, detail="Research job not found")
        
        job = agentic_research_jobs[research_id]
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Research not completed yet")
        
        results = job["results"]
        project_name = request.get("project_name", f"Agentic Research: {results.get('research_topic', 'Unknown Topic')}")
        
        # Create new project
        project = Project(
            name=project_name,
            user_id=current_user.id,
            description=f"Auto-generated from agentic research on: {results.get('research_topic', 'Unknown Topic')}"
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        
        # Create individual documents for each paper
        papers = results.get("papers", [])
        documents = []
        
        print(f"🔍 Creating {len(papers)} documents for agentic research...")
        
        for i, paper in enumerate(papers):
            # Create a document for each paper
            doc = Document(
                project_id=project.id,
                filename=f"agentic_paper_{i+1}_{research_id}.txt",
                original_name=f"{paper.get('title', f'Paper {i+1}')[:100]}...",
                processed=1,
                selected=True
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            documents.append(doc)
            print(f"✅ Created document {i+1}: {doc.original_name}")
        
        # If no papers, create a single virtual document
        if not papers:
            doc = Document(
                project_id=project.id,
                filename=f"agentic_research_{research_id}.txt",
                original_name=f"Agentic Research: {results.get('research_topic', 'Unknown Topic')}",
                processed=1,
                selected=True
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            documents.append(doc)
        
        # Save the knowledge graph nodes and edges to the first document
        # (or distribute across documents if we want to split the graph)
        graph_data = results.get("knowledge_graph")
        if graph_data and hasattr(graph_data, 'nodes') and hasattr(graph_data, 'edges'):
            # For now, save all nodes and edges to the first document
            # In the future, we could distribute them based on which paper they came from
            primary_doc = documents[0]
            
            # Save nodes
            for node in graph_data.nodes:
                pdf_node = PDFGraphNode(
                    document_id=primary_doc.id,
                    entity_id=node.id,
                    entity_type=node.group.value,
                    count=node.metadata.get("count", 1),
                    degree=node.value
                )
                db.add(pdf_node)
            
            # Save edges
            for edge in graph_data.edges:
                pdf_edge = PDFGraphEdge(
                    document_id=primary_doc.id,
                    source_id=edge.source,
                    target_id=edge.target,
                    weight=edge.value,
                    evidence=edge.metadata.get("all_evidence", []),
                    relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                )
                db.add(pdf_edge)
            
            db.commit()
        
        # Download actual PDFs and process them like regular uploaded PDFs
        if papers:
            print(f"🔍 Downloading and processing {len(papers)} PDFs...")
            
            # Initialize services
            llm_service = LLMService()
            pdf_processor = PDFProcessor()
            ner_service = NERService()
            relationship_extractor = RelationshipExtractor(llm_service)
            document_chunker = DocumentChunker(chunk_size=500, overlap=100)
            rag_service = RAGService(llm_service=llm_service)
            graph_builder = GraphBuilder()
            
            # Process each paper by downloading and processing like regular PDFs
            for i, (paper, doc) in enumerate(zip(papers, documents)):
                try:
                    print(f"🔍 Processing paper {i+1}: {paper.get('title', 'Unknown')[:50]}...")
                    
                    # Try to download the actual PDF from PubMed
                    pdf_url = None
                    if paper.get('pmid'):
                        # Try to get PDF URL from PubMed
                        try:
                            import requests
                            # Use PMC or direct PDF links if available
                            if paper.get('pmc_id'):
                                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['pmc_id']}/pdf/"
                            elif paper.get('doi'):
                                # Try to get PDF from DOI
                                pdf_url = f"https://doi.org/{paper['doi']}"
                        except Exception as e:
                            print(f"⚠️ Could not get PDF URL for paper {i+1}: {e}")
                    
                    # If we have a PDF URL, try to download it
                    if pdf_url:
                        try:
                            import requests
                            response = requests.get(pdf_url, timeout=30)
                            if response.status_code == 200:
                                # Save as PDF file
                                pdf_filename = f"agentic_paper_{i+1}_{research_id}.pdf"
                                pdf_path = f"uploads/{pdf_filename}"
                                
                                with open(pdf_path, 'wb') as f:
                                    f.write(response.content)
                                
                                print(f"📄 Downloaded PDF: {pdf_filename}")
                                
                                # Process the PDF like a regular uploaded PDF
                                pdf_result = pdf_processor.process_pdf(pdf_path)
                                
                                if pdf_result and not pdf_result.get("error"):
                                    # Update document with PDF filename
                                    doc.filename = pdf_filename
                                    db.commit()
                                    
                                    # Process with NER
                                    sentences = pdf_result.get("sentences", [])
                                    full_text = " ".join(sentences)
                                    
                                    entities = ner_service.extract_entities(full_text)
                                    
                                    # Format entities for get_unique_entities method
                                    sentence_entities = [{"entities": entities}]
                                    unique_entities = ner_service.get_unique_entities(sentence_entities)
                                    
                                    # Format entities for relationship extraction
                                    sentence_entities_for_relationships = [{"entities": entities, "sentence": full_text}]
                                    relationships = relationship_extractor.extract_all_relationships(sentence_entities_for_relationships)
                                    
                                    # Build graph for this paper
                                    graph_data = graph_builder.build_graph(unique_entities, relationships)
                                    
                                    # Save nodes and edges from the graph
                                    for node in graph_data.nodes:
                                        pdf_node = PDFGraphNode(
                                            document_id=doc.id,
                                            entity_id=node.id,
                                            entity_type=node.group.value,
                                            count=node.metadata.get("count", 1),
                                            degree=node.value
                                        )
                                        db.add(pdf_node)
                                    
                                    for edge in graph_data.edges:
                                        pdf_edge = PDFGraphEdge(
                                            document_id=doc.id,
                                            source_id=edge.source,
                                            target_id=edge.target,
                                            weight=edge.value,
                                            evidence=edge.metadata.get("all_evidence", []),
                                            relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                                        )
                                        db.add(pdf_edge)
                                    
                                    # Chunk document for RAG
                                    chunks = document_chunker.chunk_with_entities(
                                        full_text,
                                        entities,
                                        doc_id=f"agentic_paper_{i+1}_{research_id}"
                                    )
                                    
                                    # Index in RAG
                                    rag_service.index_document(
                                        doc_id=f"agentic_paper_{i+1}_{research_id}",
                                        text_chunks=chunks,
                                        entities=list(unique_entities.keys())
                                    )
                                    
                                    print(f"✅ Processed PDF {i+1}: {paper.get('title', 'Unknown')[:50]}...")
                                    continue
                                    
                        except Exception as e:
                            print(f"⚠️ Could not download PDF for paper {i+1}: {e}")
                    
                    # Fallback: Create text file if PDF download failed
                    print(f"📄 Creating text file for paper {i+1} (PDF download failed)")
                    text_content = f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
                    
                    # Save as text file
                    import os
                    os.makedirs("uploads", exist_ok=True)
                    text_file_path = f"uploads/{doc.filename}"
                    with open(text_file_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    
                    # Process with NER (same as regular PDFs)
                    entities = ner_service.extract_entities(text_content)
                    
                    # Format entities for get_unique_entities method
                    sentence_entities = [{"entities": entities}]
                    unique_entities = ner_service.get_unique_entities(sentence_entities)
                    
                    # Format entities for relationship extraction
                    sentence_entities_for_relationships = [{"entities": entities, "sentence": text_content}]
                    relationships = relationship_extractor.extract_all_relationships(sentence_entities_for_relationships)
                    
                    # Build graph for this paper (same as regular PDFs)
                    graph_data = graph_builder.build_graph(unique_entities, relationships)
                    
                    # Save nodes and edges from the graph
                    for node in graph_data.nodes:
                        pdf_node = PDFGraphNode(
                            document_id=doc.id,
                            entity_id=node.id,
                            entity_type=node.group.value,
                            count=node.metadata.get("count", 1),
                            degree=node.value
                        )
                        db.add(pdf_node)
                    
                    for edge in graph_data.edges:
                        pdf_edge = PDFGraphEdge(
                            document_id=doc.id,
                            source_id=edge.source,
                            target_id=edge.target,
                            weight=edge.value,
                            evidence=edge.metadata.get("all_evidence", []),
                            relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                        )
                        db.add(pdf_edge)
                    
                    # Chunk document for RAG
                    chunks = document_chunker.chunk_document(
                        text=text_content,
                        doc_id=f"agentic_paper_{i+1}_{research_id}"
                    )
                    
                    # Index in RAG
                    rag_service.index_document(
                        doc_id=f"agentic_paper_{i+1}_{research_id}",
                        text_chunks=chunks,
                        entities=list(unique_entities.keys())
                    )
                    
                    print(f"✅ Processed text file {i+1}: {paper.get('title', 'Unknown')[:50]}...")
                    
                except Exception as e:
                    print(f"❌ Error processing agentic paper {i+1}: {e}")
                    continue
            
            # Save combined RAG index
            rag_index_path = f"uploads/{project.id}_rag_index.pkl"
            rag_service.save_index(rag_index_path)
            
            db.commit()
        
        print(f"🎉 Agentic research saved successfully!")
        print(f"📊 Project ID: {project.id}")
        print(f"📄 Documents created: {len(documents)}")
        print(f"🔗 Entities: {len(graph_data.nodes) if graph_data else 0}")
        print(f"🔗 Relationships: {len(graph_data.edges) if graph_data else 0}")
        
        return {
            "project_id": project.id,
            "project_name": project.name,
            "message": f"Agentic research saved as project: {project.name}",
            "papers_analyzed": results.get("papers_analyzed", 0),
            "documents_created": len(documents),
            "entities_count": len(graph_data.nodes) if graph_data else 0,
            "relationships_count": len(graph_data.edges) if graph_data else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving agentic research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_agentic_research(
    research_id: str,
    research_topic: str,
    max_papers: int,
    search_strategy: str,
    project_id: Optional[str]
):
    """Background task to process agentic research"""
    try:
        # Update status - Step 1: Searching
        agentic_research_jobs[research_id]["status"] = "searching"
        agentic_research_jobs[research_id]["current_stage"] = "Searching for papers"
        print(f"🔍 Step 1: Searching for papers on '{research_topic}'...")
        
        # Initialize services
        llm_service = LLMService()
        agentic_ai = AgenticAIService(llm_service)
        
        # Update progress
        agentic_research_jobs[research_id]["progress"]["papers_found"] = 0
        agentic_research_jobs[research_id]["progress"]["papers_analyzed"] = 0
        
        # Perform research with progress updates
        print(f"📚 Finding papers...")
        results = await agentic_ai.autonomous_research(
            research_topic=research_topic,
            max_papers=max_papers,
            search_strategy=search_strategy
        )
        
        papers_found = len(results.get("papers", []))
        papers_analyzed = results.get("papers_analyzed", papers_found)
        
        # Update progress IMMEDIATELY after search completes with actual values
        print(f"✅ Search phase complete: {papers_found} papers found")
        agentic_research_jobs[research_id]["progress"]["papers_found"] = papers_found
        agentic_research_jobs[research_id]["progress"]["papers_analyzed"] = papers_analyzed
        agentic_research_jobs[research_id]["current_stage"] = f"Found {papers_found} papers - Processing..."
        agentic_research_jobs[research_id]["status"] = "analyzing"
        
        # Small delay to let frontend see the papers_found update
        await asyncio.sleep(0.3)
        
        # Update status - Step 2: Extracting
        agentic_research_jobs[research_id]["status"] = "extracting"
        agentic_research_jobs[research_id]["current_stage"] = "Extracting entities and relationships"
        print(f"🧬 Step 2: Extracting entities and relationships...")
        
        # Update progress - entities extracted
        knowledge_graph = results.get("knowledge_graph")
        entities_found = len(knowledge_graph.nodes) if knowledge_graph else 0
        relationships_found = len(knowledge_graph.edges) if knowledge_graph else 0
        
        agentic_research_jobs[research_id]["progress"]["entities_extracted"] = entities_found
        agentic_research_jobs[research_id]["progress"]["relationships_found"] = relationships_found
        agentic_research_jobs[research_id]["current_stage"] = f"Extracted {entities_found} entities, {relationships_found} relationships"
        print(f"✅ Extracted {entities_found} entities and {relationships_found} relationships")
        
        # Small delay to let frontend see the entities update
        await asyncio.sleep(0.3)
        
        # Update status - Step 3: Building graph
        agentic_research_jobs[research_id]["status"] = "building"
        agentic_research_jobs[research_id]["current_stage"] = "Building knowledge graph"
        print(f"📊 Step 3: Building knowledge graph with {entities_found} entities...")
        
        # Update status - Step 4: Saving


        agentic_research_jobs[research_id]["status"] = "saving"
        agentic_research_jobs[research_id]["current_stage"] = "Saving to project"
        print(f"💾 Step 5: Saving research to project...")
        
        # Save results
        agentic_research_jobs[research_id]["results"] = results
        agentic_research_jobs[research_id]["status"] = "completed"
        agentic_research_jobs[research_id]["current_stage"] = "Research complete!"
        
        print(f"✅ Agentic research completed: {research_id}")
        print(f"   Papers analyzed: {results.get('papers_analyzed', 0)}")
        print(f"   Entities found: {entities_found}")
        print(f"   Relationships: {relationships_found}")
        
        # Automatically save the research to the current project
        try:
            print(f"💾 Auto-saving research to current project...")
            
            # Get database session
            db = SessionLocal()
            
            # Use the provided project_id or get the most recent project
            if project_id:
                project = db.query(Project).filter(Project.id == project_id).first()
            else:
                # Get the most recent project
                project = db.query(Project).order_by(Project.created_at.desc()).first()
            
            if not project:
                print(f"❌ No project found to save research to")
                return
            
            print(f"📁 Saving to project: {project.name}")
            
            # Create individual documents for each paper
            papers = results.get("papers", [])
            documents = []
            
            print(f"🔍 Creating {len(papers)} documents for agentic research...")
            
            for i, paper in enumerate(papers):
                # Use paper title as filename (truncated to fit database constraints)
                paper_title = paper.get('title', f'Paper {i+1}')
                # Sanitize filename by removing/replacing special characters
                import re
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', paper_title)
                safe_filename = f"{safe_title[:50]}_{i+1}.txt"  # Truncate and add index
                
                # Create a document for each paper
                doc = Document(
                    id=str(uuid.uuid4()),
                    project_id=project.id,
                    filename=safe_filename,
                    file_path=f"uploads/agentic_paper_{i+1}_{research_id}.txt",
                    processed=1,
                    selected=True
                )
                db.add(doc)
                db.commit()
                db.refresh(doc)
                documents.append(doc)
                print(f"✅ Created document {i+1}: {safe_filename}")
            
            # Process papers (download PDFs and process them)
            # NOTE: The system attempts to download full PDFs from PubMed Central (PMC) when available.
            # If a PDF is not accessible (paywalled or not in PMC), it falls back to processing the abstract.
            # This ensures we always get some content even when full PDFs aren't available.
            if papers:
                print(f"🔍 Downloading and processing {len(papers)} PDFs...")
                print(f"   Note: Will attempt PDF download from PMC, fallback to abstracts if unavailable")
                
                # Initialize services
                llm_service = LLMService()
                pdf_processor = PDFProcessor()
                ner_service = NERService()
                relationship_extractor = RelationshipExtractor()
                document_chunker = DocumentChunker(chunk_size=500, overlap=100)
                rag_service = RAGService(llm_service=llm_service)
                graph_builder = GraphBuilder()
                
                # Process each paper by downloading and processing like regular PDFs
                for i, (paper, doc) in enumerate(zip(papers, documents)):
                    try:
                        print(f"🔍 Processing paper {i+1}: {paper.get('title', 'Unknown')[:50]}...")
                        
                        # Try to download the actual PDF from multiple sources
                        pdf_url = None
                        pdf_downloaded = False
                        pdf_source = None
                        
                        if paper.get('pmid'):
                            # Priority 1: Google Scholar PDF (often has open access versions)
                            if paper.get('pdf_url') and paper.get('pdf_source') == 'google_scholar':
                                pdf_url = paper['pdf_url']
                                pdf_source = 'Google Scholar'
                                print(f"   📎 Found PDF on Google Scholar - attempting download...")
                            # Priority 2: PMC (PubMed Central)
                            elif paper.get('pmc_id'):
                                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{paper['pmc_id']}/pdf/"
                                pdf_source = 'PubMed Central'
                                print(f"   📎 Found PMC ID: {paper['pmc_id']} - attempting PDF download...")
                            # Priority 3: DOI
                            elif paper.get('doi'):
                                pdf_url = f"https://doi.org/{paper['doi']}"
                                pdf_source = 'DOI'
                                print(f"   📎 Found DOI: {paper['doi']} - attempting PDF download...")
                            # Priority 4: Direct PDF URL from paper metadata
                            elif paper.get('pdf_url'):
                                pdf_url = paper['pdf_url']
                                pdf_source = 'Direct Link'
                                print(f"   📎 Found direct PDF link - attempting download...")
                            else:
                                print(f"   ⚠️  No PDF sources found - will use abstract only")
                        
                        # If we have a PDF URL, try to download it
                        if pdf_url:
                            try:
                                import requests
                                print(f"   ⬇️  Downloading PDF from {pdf_source}: {pdf_url}")
                                
                                # Different strategies for different sources
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                                    'Accept': 'application/pdf,*/*'
                                }
                                
                                response = requests.get(pdf_url, timeout=30, headers=headers, allow_redirects=True)
                                
                                # Check if we got a PDF
                                content_type = response.headers.get('Content-Type', '').lower()
                                is_pdf = 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf')
                                
                                if response.status_code == 200 and (is_pdf or len(response.content) > 10000):
                                    # Save as PDF file
                                    pdf_filename = f"agentic_paper_{i+1}_{research_id}.pdf"
                                    pdf_path = f"uploads/{pdf_filename}"
                                    
                                    with open(pdf_path, 'wb') as f:
                                        f.write(response.content)
                                    
                                    # Verify it's actually a PDF by checking magic bytes
                                    if response.content[:4] == b'%PDF':
                                        pdf_downloaded = True
                                        print(f"   ✅ Successfully downloaded PDF from {pdf_source}: {pdf_filename} ({len(response.content)} bytes)")
                                        
                                        # Process the PDF like a regular uploaded PDF
                                        pdf_result = pdf_processor.process_pdfs([pdf_path])[0]
                                    
                                    if pdf_result and not pdf_result.get("error"):
                                        # Update document with PDF filename
                                        doc.filename = pdf_filename
                                        doc.file_path = pdf_path
                                        db.commit()
                                        
                                        print(f"   📖 Extracting text from PDF...")
                                        # Process with NER (same aggressive filtering as regular PDFs)
                                        sentences = pdf_result.get("sentences", [])
                                        full_text = " ".join(sentences)
                                        
                                        print(f"   🧬 Extracting entities and relationships...")
                                        # Use same NER pipeline as regular PDFs
                                        sentence_entities = ner_service.extract_entities_from_sentences(sentences)
                                        filtered_entities = ner_service.filter_entities(sentence_entities)
                                        unique_entities = ner_service.get_unique_entities(filtered_entities)
                                        
                                        # Format entities for relationship extraction
                                        relationships = relationship_extractor.extract_all_relationships(filtered_entities)
                                        
                                        print(f"   📊 Building knowledge graph...")
                                        # Build graph for this paper
                                        graph_data = graph_builder.build_graph(unique_entities, relationships)
                                        
                                        # Save nodes and edges from the graph
                                        for node in graph_data.nodes:
                                            pdf_node = PDFGraphNode(
                                                document_id=doc.id,
                                                entity_id=node.id,
                                                entity_type=node.group.value,
                                                count=node.metadata.get("count", 1),
                                                degree=node.value
                                            )
                                            db.add(pdf_node)
                                        
                                        for edge in graph_data.edges:
                                            pdf_edge = PDFGraphEdge(
                                                document_id=doc.id,
                                                source_id=edge.source,
                                                target_id=edge.target,
                                                weight=edge.value,
                                                evidence=edge.metadata.get("all_evidence", []),
                                                relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                                            )
                                            db.add(pdf_edge)
                                        
                                        # Chunk document for RAG (use filtered entities)
                                        entity_list = []
                                        for sent_data in filtered_entities:
                                            for ent in sent_data["entities"]:
                                                entity_list.append({
                                                    "text": ent["text"],
                                                    "start": ent.get("start", 0),
                                                    "end": ent.get("end", 0),
                                                    "type": ent.get("type", "ENTITY")
                                                })
                                        
                                        chunks = document_chunker.chunk_with_entities(
                                            full_text,
                                            f"agentic_paper_{i+1}_{research_id}",
                                            entity_list
                                        )
                                        
                                        # Index in RAG
                                        rag_service.index_document(
                                            doc_id=f"agentic_paper_{i+1}_{research_id}",
                                            text_chunks=chunks,
                                            entities=list(unique_entities.keys())
                                        )
                                        
                                        print(f"   ✅ Successfully processed FULL PDF {i+1} from {pdf_source}: {len(graph_data.nodes)} entities, {len(graph_data.edges)} relationships")
                                        continue
                                    else:
                                        print(f"   ⚠️  PDF processing failed, falling back to abstract")
                                else:
                                    print(f"   ⚠️  Downloaded file is not a valid PDF, falling back to abstract")
                                    # Clean up invalid file
                                    import os
                                    if os.path.exists(pdf_path):
                                        os.remove(pdf_path)
                                        
                            except Exception as e:
                                print(f"   ⚠️  Could not download/process PDF for paper {i+1}: {e}")
                                print(f"   📄 Falling back to abstract-only processing")
                        
                        # Fallback: Create text file if PDF download failed
                        if not pdf_downloaded:
                            print(f"   📄 Processing abstract only for paper {i+1}")
                        text_content = f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
                        
                        # Save as text file
                        import os
                        os.makedirs("uploads", exist_ok=True)
                        text_file_path = f"uploads/{doc.filename}"
                        with open(text_file_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        
                        print(f"   📖 Extracting entities from abstract...")
                        # Process with NER (same aggressive filtering as regular PDFs)
                        # Split text into sentences for proper NER processing
                        spacy_doc = ner_service.nlp(text_content)
                        sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
                        
                        # Use same NER pipeline as regular PDFs
                        sentence_entities = ner_service.extract_entities_from_sentences(sentences)
                        filtered_entities = ner_service.filter_entities(sentence_entities)
                        unique_entities = ner_service.get_unique_entities(filtered_entities)
                        
                        # Format entities for relationship extraction
                        relationships = relationship_extractor.extract_all_relationships(filtered_entities)
                        
                        print(f"   📊 Building knowledge graph from abstract...")
                        # Build graph for this paper (same as regular PDFs)
                        graph_data = graph_builder.build_graph(unique_entities, relationships)
                        
                        # Save nodes and edges from the graph
                        for node in graph_data.nodes:
                            pdf_node = PDFGraphNode(
                                document_id=doc.id,
                                entity_id=node.id,
                                entity_type=node.group.value,
                                count=node.metadata.get("count", 1),
                                degree=node.value
                            )
                            db.add(pdf_node)
                        
                        for edge in graph_data.edges:
                            pdf_edge = PDFGraphEdge(
                                document_id=doc.id,
                                source_id=edge.source,
                                target_id=edge.target,
                                weight=edge.value,
                                evidence=edge.metadata.get("all_evidence", []),
                                relationship_type=edge.metadata.get("relationship_type", "CO_OCCURRENCE")
                            )
                            db.add(pdf_edge)
                        
                        # Chunk document for RAG (use filtered entities)
                        entity_list = []
                        for sent_data in filtered_entities:
                            for ent in sent_data["entities"]:
                                entity_list.append({
                                    "text": ent["text"],
                                    "start": ent.get("start", 0),
                                    "end": ent.get("end", 0),
                                    "type": ent.get("type", "ENTITY")
                                })
                        
                        chunks = document_chunker.chunk_with_entities(
                            text_content,
                            f"agentic_paper_{i+1}_{research_id}",
                            entity_list
                        )
                        
                        # Index in RAG
                        rag_service.index_document(
                            doc_id=f"agentic_paper_{i+1}_{research_id}",
                            text_chunks=chunks,
                            entities=list(unique_entities.keys())
                        )
                        
                        print(f"   ✅ Successfully processed abstract {i+1}: {len(graph_data.nodes)} entities, {len(graph_data.edges)} relationships")
                        
                    except Exception as e:
                        print(f"❌ Error processing agentic paper {i+1}: {e}")
                        continue
                
                # Save combined RAG index
                rag_index_path = f"uploads/{project.id}_rag_index.pkl"
                rag_service.save_index(rag_index_path)
                
                db.commit()
            
            print(f"🎉 Agentic research added to project: {project.name}")
            print(f"📊 Project ID: {project.id}")
            print(f"📄 Documents added: {len(documents)}")
            
            # Close database session
            db.close()
            
        except Exception as e:
            print(f"❌ Error auto-saving research: {e}")
            if 'db' in locals():
                db.close()
        
    except Exception as e:
        print(f"❌ Agentic research failed: {e}")
        agentic_research_jobs[research_id]["status"] = "failed"
        agentic_research_jobs[research_id]["error"] = str(e)


async def expand_agentic_research(research_id: str, max_new_papers: int):
    """Background task to expand research with related papers"""
    try:
        job = agentic_research_jobs[research_id]
        job["status"] = "expanding"
        
        # Initialize services
        llm_service = LLMService()
        agentic_ai = AgenticAIService(llm_service)
        
        # Find related papers
        current_papers = job["results"]["papers"]
        related_papers = await agentic_ai.find_related_papers(
            current_papers=current_papers,
            max_new_papers=max_new_papers
        )
        
        # Update results with new papers
        job["results"]["papers"].extend(related_papers)
        job["results"]["papers_analyzed"] = len(job["results"]["papers"])
        
        job["status"] = "completed"
        
        print(f"✅ Research expanded: {research_id} with {len(related_papers)} new papers")
        
    except Exception as e:
        print(f"❌ Research expansion failed: {e}")
        agentic_research_jobs[research_id]["status"] = "failed"
        agentic_research_jobs[research_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )

