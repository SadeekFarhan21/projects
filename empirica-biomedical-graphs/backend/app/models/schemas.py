from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class EntityType(str, Enum):
    """Biomedical entity types from scispaCy"""
    ENTITY = "ENTITY"  # Generic biomedical entity
    GENE_OR_GENE_PRODUCT = "GENE_OR_GENE_PRODUCT"
    CHEMICAL = "CHEMICAL"
    DISEASE = "DISEASE"
    ORGANISM = "ORGANISM"
    CELL_TYPE = "CELL_TYPE"
    TISSUE = "TISSUE"
    ORGAN = "ORGAN"
    UNKNOWN = "UNKNOWN"


class Node(BaseModel):
    """Graph node representing a biomedical entity"""
    id: str = Field(..., description="Unique identifier (entity name)")
    group: EntityType = Field(..., description="Entity type/category")
    value: int = Field(default=1, description="Node size (degree/importance)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")


class Edge(BaseModel):
    """Graph edge representing a relationship between entities"""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    value: float = Field(default=1.0, description="Relationship strength/weight")
    title: str = Field(default="", description="Evidence sentence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")


class GraphData(BaseModel):
    """Complete graph structure for visualization"""
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph-level metadata")


class GraphAnalytics(BaseModel):
    """Graph analytics and statistics"""
    total_nodes: int
    total_edges: int
    density: float
    avg_degree: float
    communities: List[List[str]]
    centrality_scores: Dict[str, float]
    entity_counts: Dict[str, int]


class ProcessingStatus(BaseModel):
    """Status of PDF processing job"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str = ""
    result: Optional[GraphData] = None


class PDFMetadata(BaseModel):
    """Metadata for an individual PDF and its graph"""
    document_id: str
    filename: str
    uploaded_at: str
    processed: bool
    selected: bool
    node_count: int
    edge_count: int
    entity_counts: Dict[str, int] = {}


class ProjectMetadata(BaseModel):
    """Project information"""
    project_id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: str
    pdf_count: int
    pdfs: List[PDFMetadata] = []


# ==== Conversational Agent Schemas ====

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    message: str
    graph: GraphData
    conversation_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    citations: List[str] = []
    relevant_nodes: List[str] = []
    relevant_edges: List[List[str]] = []  # [[source, target], ...]
    tool_calls: List[str] = []


# ==== Hypothesis Generation Schemas ====

class Hypothesis(BaseModel):
    title: str
    explanation: str
    entities: List[str]
    evidence_sentences: List[str] = []
    edge_pairs: List[List[str]] = []  # supporting edges [[a,b], [b,c]]
    confidence: float = 0.5


class HypothesesRequest(BaseModel):
    graph: GraphData
    focus_entity: Optional[str] = None
    max_results: int = 10


class HypothesesResponse(BaseModel):
    hypotheses: List[Hypothesis]


# ==== NER Preview Schemas ====

class NerPreviewRequest(BaseModel):
    text: str
    min_occurrences: int = 2
    return_raw: bool = False


class NerSentenceEntities(BaseModel):
    sentence_id: int
    sentence: str
    entities: List[Dict[str, Any]]


class NerPreviewResponse(BaseModel):
    sentences: List[NerSentenceEntities]
    unique_entities: Dict[str, Dict[str, Any]]
    raw_sentences: Optional[List[NerSentenceEntities]] = None
    debug: Optional[Dict[str, Any]] = None


# ==== Project Export/Import Schemas ====

class PDFGraphExport(BaseModel):
    """Export format for a single PDF's graph"""
    document_id: str
    filename: str
    uploaded_at: str
    graph: GraphData


class ProjectExport(BaseModel):
    """Export format for entire project with all PDF graphs"""
    project_name: str
    project_id: str
    created_at: str
    updated_at: str
    pdf_graphs: List[PDFGraphExport]
    settings: Dict[str, Any] = {}


class ProjectImportRequest(BaseModel):
    project_data: ProjectExport
    merge_with_existing: bool = False


class PDFSelectionRequest(BaseModel):
    """Request to update which PDFs are selected for graph visualization"""
    project_id: str
    selected_document_ids: List[str]


# ==== Paper Discovery Schemas ====

class PaperDiscoveryRequest(BaseModel):
    query: str
    max_results: int = 10
    auto_merge: bool = True
    source: str = "pubmed"  # pubmed, arxiv, etc


class DiscoveredPaper(BaseModel):
    id: str  # PMID or DOI
    title: str
    abstract: str
    authors: List[str] = []
    journal: str = ""
    year: Optional[int] = None
    url: str = ""


class PaperDiscoveryResponse(BaseModel):
    papers: List[DiscoveredPaper]
    job_id: Optional[str] = None
    status: str = "completed"


# ==== Clinical Trials Schemas ====

class TrialDiscoveryRequest(BaseModel):
    condition: str
    max_results: int = 20
    phases: Optional[List[str]] = None  # ["PHASE1", "PHASE2", etc]
    status: Optional[List[str]] = None  # ["RECRUITING", "COMPLETED", etc]


class ClinicalTrial(BaseModel):
    nct_id: str
    title: str
    condition: str
    interventions: List[str] = []
    phase: str = ""
    status: str = ""
    sponsor: str = ""
    brief_summary: str = ""
    url: str = ""


class TrialDiscoveryResponse(BaseModel):
    trials: List[ClinicalTrial]
    graph: Optional[GraphData] = None

