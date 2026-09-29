import spacy
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re


class NERService:
    """Named Entity Recognition for biomedical text using scispaCy"""
    
    def __init__(self, model_name: str = "en_ner_bionlp13cg_md"):
        """Initialize the NER model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise Exception(
                f"Model '{model_name}' not found. Please install it:\n"
                f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz"
            )
        
        # Entity type mappings
        # Expand mapping to cover BioNLP/BCRAFT fine-grained labels
        self.entity_type_map = {
            # Generic
            "ENTITY": "ENTITY",
            # Genes/Proteins
            "GENE": "GENE_OR_GENE_PRODUCT",
            "GENE_OR_GENE_PRODUCT": "GENE_OR_GENE_PRODUCT",
            "GENE_OR_GENE": "GENE_OR_GENE_PRODUCT",
            "PROTEIN": "GENE_OR_GENE_PRODUCT",
            "GENE_OR_GENE_PRODUCT_BIO": "GENE_OR_GENE_PRODUCT",
            # Chemicals
            "CHEMICAL": "CHEMICAL",
            "SIMPLE_CHEMICAL": "CHEMICAL",
            "AMINO_ACID": "CHEMICAL",
            "ION": "CHEMICAL",
            # Diseases/Phenomena
            "DISEASE": "DISEASE",
            "CANCER": "DISEASE",
            "PATHOLOGICAL_FORMATION": "DISEASE",
            # Organism/Anatomy
            "ORGANISM": "ORGANISM",
            "TISSUE": "TISSUE",
            "CELL": "CELL_TYPE",
            "CELL_TYPE": "CELL_TYPE",
            "CELL_LINE": "CELL_TYPE",
            "ORGAN": "ORGAN",
            # Processes/Regulation
            "BIOLOGICAL_PROCESS": "ENTITY",
            "REGULATOR": "ENTITY",
        }
        
        # Minimum entity occurrence threshold - require entities to appear multiple times
        self.min_entity_occurrences = 4
        
        # Junk patterns to exclude (case-insensitive)
        self.exclude_patterns = [
            # Academic metadata - be more specific to avoid false positives
            r'^(international journal)',
            r'(journal of|review of)',
            r'(university|institute|department of)',
            r'^(figure|fig\.|table|supplementary|appendix)',
            r'(citation:|reference:|doi:|pmid:|issn:)',
            r'(copyright|license agreement|published by)',
            
            # Author-related
            r'^\d+[,\s]*\d+[,\s]*\d+',  # Numbers like "1,2,3" or "1, 2, 3"
            r'et al\.',
            
            # Section headers
            r'^(abstract|introduction|methods|materials and methods|results|discussion|conclusion|acknowledgments)$',
            
            # Formatting artifacts
            r'[\x00-\x1f\x7f-\x9f]',  # Control characters
            r'^[^a-z0-9]+$',  # Only special chars
        ]
        
        # Common academic/non-medical exact words to skip
        self.skip_words = {
            'citation', 'figure', 'table', 'references', 'et al', 'doi',
            'abstract', 'introduction', 'methods', 'results', 'discussion',
            'conclusion', 'supplementary', 'materials', 'acknowledgments',
            'university', 'department', 'institute', 'laboratory', 'center',
            'journal', 'review', 'article', 'study',
            'copyright', 'license', 'published', 'publisher', 'corresponding',
            'author', 'authors', 'correspondence', 'affiliation', 'affiliations',
            'international journal', 'and', 'or', 'the', 'of', 'in', 'on', 'at',
        }
        # Additional skip patterns: initials, lone letters/short tokens
        self.initials_regex = re.compile(r"^(?:[A-Z]\.?){1,3}$")  # e.g., R., A., R.P.
        self.short_token_regex = re.compile(r"^[A-Za-z]{1,2}$")
    
    def _is_valid_biomedical_entity(self, text: str, label: str) -> bool:
        """Determine if an entity is a valid biomedical entity"""
        text_lower = text.lower().strip()
        
        # Length checks
        if len(text) < 3 or len(text) > 80:
            return False
        
        # Check skip words
        if text_lower in self.skip_words:
            return False
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Must have at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Skip if it's mostly numbers or special characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.4:
            return False
        
        # Skip initials and very short tokens
        if self.initials_regex.match(text.strip()) or self.short_token_regex.match(text.strip()):
            return False

        # ONLY accept mapped biomedical entity types
        if label not in self.entity_type_map:
            return False
        
        return True
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """Extract entities from text"""
        doc = self.nlp(text)
        entities = []
        filtered_count = {"total": 0, "by_label": defaultdict(int), "by_reason": defaultdict(int)}
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            filtered_count["total"] += 1
            filtered_count["by_label"][ent.label_] += 1
            
            # Apply strict biomedical filtering
            if not self._is_valid_biomedical_entity(entity_text, ent.label_):
                filtered_count["by_reason"]["filtered_out"] += 1
                continue
            
            # Only use mapped entity types
            entity_type = self.entity_type_map.get(ent.label_)
            if entity_type:
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
                filtered_count["by_reason"]["accepted"] += 1
        
        return entities
    
    def extract_entities_from_sentences(self, sentences: List[str]) -> List[Dict[str, any]]:
        """Extract entities from a list of sentences"""
        results = []
        all_labels_seen = set()
        sample_entities_by_label = defaultdict(list)
        
        # Debug: sample first few sentences
        if len(sentences) > 0:
            print(f"DEBUG NER: First sentence sample: {sentences[0][:200]}")
        
        # Process ALL sentences (but collect samples for debugging)
        for idx, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            
            # Collect label statistics from raw output
            for ent in doc.ents:
                all_labels_seen.add(ent.label_)
                if len(sample_entities_by_label[ent.label_]) < 3:
                    sample_entities_by_label[ent.label_].append(ent.text[:50])
            
            entities = self.extract_entities(sentence)
            if idx < 3 and entities:
                print(f"DEBUG NER: Sentence {idx} found {len(entities)} entities")
                print(f"DEBUG NER: Entity examples: {[e['text'] for e in entities[:5]]}")
            if entities:
                results.append({
                    "sentence_id": idx,
                    "sentence": sentence,
                    "entities": entities
                })
        
        print(f"\n{'='*60}")
        print(f"DEBUG NER: Entity Label Analysis")
        print(f"{'='*60}")
        print(f"All entity labels found by scispaCy: {sorted(all_labels_seen)}")
        print(f"\nAccepted labels (mapped): {list(self.entity_type_map.keys())}")
        print(f"\nSample entities by label:")
        for label in sorted(sample_entities_by_label.keys()):
            is_accepted = label in self.entity_type_map
            status = "✓ ACCEPTED" if is_accepted else "✗ REJECTED"
            print(f"  {label:20} {status:12} - {sample_entities_by_label[label]}")
        print(f"\nProcessed {len(sentences)} sentences, found entities in {len(results)} sentences")
        print(f"{'='*60}\n")
        return results
    
    def get_entity_counts(self, sentence_entities: List[Dict[str, any]]) -> Dict[str, int]:
        """Count entity occurrences across all sentences"""
        entity_counts = defaultdict(int)
        
        for sent_data in sentence_entities:
            seen_in_sentence = set()
            for entity in sent_data["entities"]:
                entity_name = self._normalize_entity(entity["text"])
                seen_in_sentence.add(entity_name)
            
            for entity_name in seen_in_sentence:
                entity_counts[entity_name] += 1
        
        return dict(entity_counts)
    
    def filter_entities(self, sentence_entities: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Filter out low-frequency entities"""
        entity_counts = self.get_entity_counts(sentence_entities)
        
        filtered_results = []
        for sent_data in sentence_entities:
            filtered_entities = [
                entity for entity in sent_data["entities"]
                if entity_counts.get(self._normalize_entity(entity["text"]), 0) >= self.min_entity_occurrences
            ]
            
            if filtered_entities:
                filtered_results.append({
                    "sentence_id": sent_data["sentence_id"],
                    "sentence": sent_data["sentence"],
                    "entities": filtered_entities
                })
        
        return filtered_results
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for comparison"""
        # Remove extra whitespace, lowercase, remove special chars
        text = re.sub(r'\s+', ' ', text.strip())
        return text.lower()
    
    def get_unique_entities(self, sentence_entities: List[Dict[str, any]]) -> Dict[str, Dict[str, any]]:
        """Get unique entities with their types and occurrence counts"""
        entities = {}
        
        for sent_data in sentence_entities:
            for entity in sent_data["entities"]:
                entity_name = entity["text"]
                normalized = self._normalize_entity(entity_name)
                
                if normalized not in entities:
                    entities[normalized] = {
                        "original_name": entity_name,
                        "type": entity["type"],
                        "count": 0
                    }
                
                entities[normalized]["count"] += 1
        
        return entities

