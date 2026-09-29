from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import json
import networkx as nx
from pathlib import Path
import pickle


class RAGService:
    """
    Retrieval-Augmented Generation service that combines:
    - Vector embeddings of PDF content
    - Graph-based relationship context
    - Semantic search for relevant chunks
    - Entity-aware retrieval
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        
        # Storage for document chunks and embeddings
        self.document_chunks: Dict[str, List[Dict[str, Any]]] = {}  # doc_id -> chunks
        self.chunk_embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
        self.entity_to_chunks: Dict[str, List[str]] = defaultdict(list)  # entity -> chunk_ids
        
        # Graph context
        self.graph: Optional[nx.Graph] = None
        self.entity_metadata: Dict[str, Dict] = {}
        
    def index_document(
        self, 
        doc_id: str, 
        text_chunks: List[Dict[str, Any]], 
        entities: List[Dict[str, Any]]
    ):
        """
        Index a document by:
        1. Storing text chunks
        2. Creating embeddings (if LLM available)
        3. Linking entities to chunks
        
        Args:
            doc_id: Document identifier
            text_chunks: List of text chunks with metadata
                [{"text": str, "page": int, "chunk_id": str, "entities": List[str]}]
            entities: List of entities found in the document
        """
        self.document_chunks[doc_id] = text_chunks
        
        # Link entities to chunks
        for chunk in text_chunks:
            chunk_id = chunk.get("chunk_id")
            chunk_entities = chunk.get("entities", [])
            
            for entity in chunk_entities:
                self.entity_to_chunks[entity].append(chunk_id)
        
        # TODO: Generate embeddings if LLM service is available
        # This would use Anthropic embeddings or similar
        
    def set_graph_context(self, graph: nx.Graph, entity_metadata: Dict[str, Dict]):
        """Set the knowledge graph for graph-enhanced retrieval"""
        self.graph = graph
        self.entity_metadata = entity_metadata
        
    def retrieve_context_for_query(
        self, 
        query: str, 
        entities: List[str] = None,
        top_k: int = 5,
        include_graph_context: bool = True,
        target_doc_id: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query using hybrid approach:
        1. Entity-based retrieval (if entities provided)
        2. Semantic search (if embeddings available)
        3. Graph-based expansion (if graph available)
        
        Returns:
            {
                "chunks": List[Dict],  # Relevant text chunks
                "entities": List[str],  # Related entities
                "relationships": List[Tuple],  # Relevant relationships
                "graph_context": str,  # Summary of graph relationships
            }
        """
        relevant_chunks = []
        relevant_entities = set(entities or [])
        relevant_relationships = []
        
        # 1. Entity-based retrieval
        if entities:
            chunk_scores = defaultdict(float)
            
            for entity in entities:
                # Get chunks mentioning this entity
                chunk_ids = self.entity_to_chunks.get(entity, [])
                for chunk_id in chunk_ids:
                    chunk_scores[chunk_id] += 1.0
                
                # If graph available, expand to related entities
                if self.graph and entity in self.graph.nodes():
                    neighbors = list(self.graph.neighbors(entity))
                    relevant_entities.update(neighbors[:5])  # Add top 5 neighbors
                    
                    # Get relationships
                    for neighbor in neighbors[:3]:
                        edge_data = self.graph.get_edge_data(entity, neighbor, {})
                        relevant_relationships.append({
                            "source": entity,
                            "target": neighbor,
                            "type": edge_data.get("relationship_type", "RELATED"),
                            "weight": edge_data.get("weight", 1.0),
                            "evidence": edge_data.get("evidence", "")
                        })
            
            # Get top chunks by score
            top_chunk_ids = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for chunk_id, score in top_chunk_ids:
                # Find the chunk in document_chunks
                for doc_id, chunks in self.document_chunks.items():
                    # Filter by target document if specified
                    if target_doc_id and doc_id != target_doc_id:
                        continue
                        
                    for chunk in chunks:
                        if chunk.get("chunk_id") == chunk_id:
                            relevant_chunks.append({
                                **chunk,
                                "relevance_score": score,
                                "doc_id": doc_id
                            })
                            break
        
        # 2. Graph context summary
        graph_context = ""
        if include_graph_context and self.graph and entities:
            graph_context = self._build_graph_context_summary(list(relevant_entities))
        
        return {
            "chunks": relevant_chunks[:top_k],
            "entities": list(relevant_entities),
            "relationships": relevant_relationships,
            "graph_context": graph_context,
            "total_chunks_available": sum(len(chunks) for chunks in self.document_chunks.values())
        }
    
    def _build_graph_context_summary(self, entities: List[str]) -> str:
        """Build a natural language summary of graph relationships"""
        if not self.graph or not entities:
            return ""
        
        summary_parts = []
        
        # Get subgraph of relevant entities
        relevant_nodes = set()
        for entity in entities:
            if entity in self.graph.nodes():
                relevant_nodes.add(entity)
                # Add immediate neighbors
                relevant_nodes.update(list(self.graph.neighbors(entity))[:3])
        
        # Describe key relationships
        for entity in entities[:5]:  # Limit to top 5 entities
            if entity not in self.graph.nodes():
                continue
                
            neighbors = list(self.graph.neighbors(entity))
            if neighbors:
                # Get strongest connections
                connections = []
                for neighbor in neighbors[:3]:
                    edge_data = self.graph.get_edge_data(entity, neighbor, {})
                    weight = edge_data.get("weight", 1.0)
                    rel_type = edge_data.get("relationship_type", "related to")
                    connections.append(f"{neighbor} ({rel_type}, strength: {weight:.2f})")
                
                if connections:
                    summary_parts.append(
                        f"{entity} is connected to: {', '.join(connections)}"
                    )
        
        return "\n".join(summary_parts)
    
    def build_rag_prompt(
        self, 
        query: str, 
        context: Dict[str, Any],
        task_type: str = "answer"
    ) -> str:
        """
        Build a comprehensive RAG prompt with retrieved context
        
        Args:
            query: User query
            context: Retrieved context from retrieve_context_for_query
            task_type: Type of task (answer, hypothesis, summary, etc.)
        """
        chunks = context.get("chunks", [])
        entities = context.get("entities", [])
        relationships = context.get("relationships", [])
        graph_context = context.get("graph_context", "")
        
        if task_type == "answer":
            prompt = f"""You are a biomedical research assistant. Answer the following question using the provided context from research papers and knowledge graph.

QUESTION: {query}

RELEVANT TEXT EXCERPTS:
"""
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get("text", "")
                doc_id = chunk.get("doc_id", "unknown")
                page = chunk.get("page", "?")
                prompt += f"\n[Excerpt {i} from Document {doc_id}, Page {page}]\n{text}\n"
            
            if graph_context:
                prompt += f"\nKNOWLEDGE GRAPH CONTEXT:\n{graph_context}\n"
            
            if relationships:
                prompt += "\nKEY RELATIONSHIPS:\n"
                for rel in relationships[:5]:
                    prompt += f"- {rel['source']} {rel['type']} {rel['target']} (confidence: {rel['weight']:.2f})\n"
            
            prompt += f"\nRELEVANT ENTITIES: {', '.join(entities[:10])}\n"
            prompt += "\nINSTRUCTIONS:\n"
            prompt += "1. Answer the question based on the provided context\n"
            prompt += "2. Cite specific excerpts when making claims\n"
            prompt += "3. If the context doesn't contain enough information, say so\n"
            prompt += "4. Be specific and scientific in your response\n"
            prompt += "\nANSWER:"
            
        elif task_type == "hypothesis":
            prompt = f"""You are a biomedical research analyst. Generate research hypotheses based on the following data.

RESEARCH FOCUS: {query}

RELEVANT TEXT EXCERPTS:
"""
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get("text", "")
                prompt += f"\n[Excerpt {i}]\n{text}\n"
            
            if graph_context:
                prompt += f"\nKNOWLEDGE GRAPH CONTEXT:\n{graph_context}\n"
            
            prompt += "\nGENERATE 5-8 RESEARCH HYPOTHESES that:\n"
            prompt += "1. Are novel and testable\n"
            prompt += "2. Connect multiple entities or concepts\n"
            prompt += "3. Have clear clinical or research implications\n"
            prompt += "4. Are supported by the provided context\n"
            prompt += "\nReturn as JSON array with: title, description, entities, evidence, confidence, implications\n"
            
        elif task_type == "summary":
            prompt = f"""Summarize the key findings about: {query}

CONTEXT:
"""
            for chunk in chunks:
                prompt += f"\n{chunk.get('text', '')}\n"
            
            if graph_context:
                prompt += f"\nRelationships:\n{graph_context}\n"
            
            prompt += "\nProvide a concise summary of the key findings, mechanisms, and implications."
        
        return prompt
    
    def get_entity_context(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get comprehensive context for a specific entity including:
        - Text chunks mentioning the entity
        - Graph neighborhood
        - Related entities and relationships
        """
        context = {
            "entity": entity,
            "chunks": [],
            "neighbors": [],
            "relationships": [],
            "metadata": self.entity_metadata.get(entity, {})
        }
        
        # Get chunks mentioning this entity
        chunk_ids = self.entity_to_chunks.get(entity, [])
        for chunk_id in chunk_ids:
            for doc_id, chunks in self.document_chunks.items():
                for chunk in chunks:
                    if chunk.get("chunk_id") == chunk_id:
                        context["chunks"].append({**chunk, "doc_id": doc_id})
        
        # Get graph neighborhood
        if self.graph and entity in self.graph.nodes():
            neighbors = list(self.graph.neighbors(entity))
            context["neighbors"] = neighbors
            
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(entity, neighbor, {})
                context["relationships"].append({
                    "target": neighbor,
                    "type": edge_data.get("relationship_type", "RELATED"),
                    "weight": edge_data.get("weight", 1.0),
                    "evidence": edge_data.get("evidence", "")
                })
        
        return context
    
    def save_index(self, filepath: str):
        """Save the RAG index to disk"""
        index_data = {
            "document_chunks": self.document_chunks,
            "chunk_embeddings": self.chunk_embeddings,
            "entity_to_chunks": dict(self.entity_to_chunks),
            "entity_metadata": self.entity_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, filepath: str):
        """Load the RAG index from disk"""
        if not Path(filepath).exists():
            return False
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.document_chunks = index_data.get("document_chunks", {})
        self.chunk_embeddings = index_data.get("chunk_embeddings", {})
        self.entity_to_chunks = defaultdict(list, index_data.get("entity_to_chunks", {}))
        self.entity_metadata = index_data.get("entity_metadata", {})
        
        return True

