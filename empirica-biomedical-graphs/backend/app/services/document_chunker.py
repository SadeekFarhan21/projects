from typing import List, Dict, Any
import re
import hashlib


class DocumentChunker:
    """
    Intelligent document chunking that:
    - Splits text into semantic chunks
    - Preserves sentence boundaries
    - Maintains entity context
    - Tracks metadata (page, position, etc.)
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Args:
            chunk_size: Target size for chunks (in characters)
            overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(
        self, 
        text: str, 
        doc_id: str,
        page_boundaries: List[int] = None,
        entities_per_page: Dict[int, List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into overlapping segments
        
        Args:
            text: Full document text
            doc_id: Document identifier
            page_boundaries: Character positions where pages start
            entities_per_page: Entities found on each page
            
        Returns:
            List of chunks with metadata
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = self._generate_chunk_id(doc_id, chunk_idx)
                
                # Determine page number
                page_num = self._get_page_number(
                    text.find(chunk_text), 
                    page_boundaries
                )
                
                # Get entities in this chunk
                chunk_entities = self._extract_entities_from_chunk(
                    chunk_text,
                    entities_per_page.get(page_num, []) if entities_per_page else []
                )
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_text),
                    "entities": chunk_entities
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
                chunk_idx += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_chunk_id(doc_id, chunk_idx)
            page_num = self._get_page_number(
                text.find(chunk_text), 
                page_boundaries
            )
            
            chunk_entities = self._extract_entities_from_chunk(
                chunk_text,
                entities_per_page.get(page_num, []) if entities_per_page else []
            )
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "doc_id": doc_id,
                "page": page_num,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_text),
                "entities": chunk_entities
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (can be improved with spaCy)
        # Split on period, exclamation, question mark followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        overlap_chars = 0
        overlap_sentences = []
        
        # Get sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if overlap_chars >= self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_chars += len(sentence)
        
        return overlap_sentences
    
    def _generate_chunk_id(self, doc_id: str, chunk_idx: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_id}_chunk_{chunk_idx}"
    
    def _get_page_number(
        self, 
        char_position: int, 
        page_boundaries: List[int] = None
    ) -> int:
        """Determine page number from character position"""
        if not page_boundaries:
            return 1
        
        for page_num, boundary in enumerate(page_boundaries, 1):
            if char_position < boundary:
                return page_num
        
        return len(page_boundaries)
    
    def _extract_entities_from_chunk(
        self, 
        chunk_text: str, 
        available_entities: List[str]
    ) -> List[str]:
        """Extract entities that appear in this chunk"""
        chunk_lower = chunk_text.lower()
        found_entities = []
        
        for entity in available_entities:
            # Check if entity appears in chunk (case-insensitive)
            if entity.lower() in chunk_lower:
                found_entities.append(entity)
        
        return found_entities
    
    def chunk_with_entities(
        self,
        text: str,
        doc_id: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk document while tracking which entities appear in each chunk
        
        Args:
            text: Document text
            doc_id: Document ID
            entities: List of entities with positions
                [{"text": str, "start": int, "end": int, "type": str}]
        """
        # Group entities by approximate position
        entities_by_position = {}
        for entity in entities:
            pos = entity.get("start", 0)
            if pos not in entities_by_position:
                entities_by_position[pos] = []
            entities_by_position[pos].append(entity.get("text", ""))
        
        # Create chunks
        chunks = self.chunk_document(text, doc_id)
        
        # Match entities to chunks
        for chunk in chunks:
            chunk_start = text.find(chunk["text"])
            chunk_end = chunk_start + len(chunk["text"])
            
            # Find entities that overlap with this chunk
            chunk_entities = set()
            for entity in entities:
                entity_start = entity.get("start", 0)
                entity_end = entity.get("end", 0)
                
                # Check if entity overlaps with chunk
                if (chunk_start <= entity_start < chunk_end) or \
                   (chunk_start < entity_end <= chunk_end) or \
                   (entity_start <= chunk_start and entity_end >= chunk_end):
                    chunk_entities.add(entity.get("text", ""))
            
            chunk["entities"] = list(chunk_entities)
        
        return chunks

