import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
from pathlib import Path


class PDFProcessor:
    """Extract and process text from PDF documents"""
    
    def __init__(self):
        self.min_sentence_length = 10
        self.max_sentence_length = 1000
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
            }
        except Exception as e:
            return {}
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)  # Fix missing spaces after periods
        
        # Split into sentences (simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter sentences
        sentences = [
            s.strip() for s in sentences 
            if self.min_sentence_length <= len(s) <= self.max_sentence_length
        ]
        
        return sentences
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """Process multiple PDFs and return structured data"""
        results = []
        
        for pdf_path in pdf_paths:
            try:
                text = self.extract_text(pdf_path)
                sentences = self.split_into_sentences(text)
                metadata = self.extract_metadata(pdf_path)
                
                results.append({
                    "filename": Path(pdf_path).name,
                    "text": text,
                    "sentences": sentences,
                    "metadata": metadata,
                    "sentence_count": len(sentences),
                    "char_count": len(text)
                })
            except Exception as e:
                results.append({
                    "filename": Path(pdf_path).name,
                    "error": str(e)
                })
        
        return results

