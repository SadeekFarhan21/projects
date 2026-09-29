from typing import List, Dict, Tuple, Set
from collections import defaultdict
from itertools import combinations
import re


class RelationshipExtractor:
    """Extract relationships between entities based on co-occurrence and patterns"""
    
    def __init__(self):
        self.min_relationship_strength = 1  # Minimum co-occurrences to keep
        
        # Relationship patterns for semantic extraction
        self.relationship_patterns = [
            # Causation
            (r"(\w+)\s+(causes?|induces?|triggers?|leads to|results in)\s+(\w+)", "CAUSES"),
            (r"(\w+)\s+(is caused by|is induced by|is triggered by)\s+(\w+)", "CAUSED_BY"),
            
            # Inhibition
            (r"(\w+)\s+(inhibits?|blocks?|suppresses?|prevents?)\s+(\w+)", "INHIBITS"),
            (r"(\w+)\s+(is inhibited by|is blocked by|is suppressed by)\s+(\w+)", "INHIBITED_BY"),
            
            # Association
            (r"(\w+)\s+(associates? with|interacts? with|binds? to)\s+(\w+)", "INTERACTS_WITH"),
            
            # Treatment
            (r"(\w+)\s+(treats?|ameliorates?|reduces?)\s+(\w+)", "TREATS"),
            
            # Expression/Regulation
            (r"(\w+)\s+(expresses?|activates?|upregulates?|downregulates?)\s+(\w+)", "REGULATES"),
        ]
    
    def extract_cooccurrence_relationships(
        self, 
        sentence_entities: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Extract relationships based on entity co-occurrence in sentences"""
        relationships = defaultdict(lambda: {"weight": 0, "evidence": []})
        
        for sent_data in sentence_entities:
            entities = sent_data["entities"]
            sentence = sent_data["sentence"]
            
            # Create pairs of entities in the same sentence
            if len(entities) >= 2:
                for i, ent1 in enumerate(entities):
                    for ent2 in entities[i+1:]:
                        # Create canonical edge (alphabetically sorted to avoid duplicates)
                        entity1 = ent1["text"]
                        entity2 = ent2["text"]
                        
                        # Skip if same entity
                        if entity1.lower() == entity2.lower():
                            continue
                        
                        edge_key = tuple(sorted([entity1, entity2]))
                        relationships[edge_key]["weight"] += 1
                        
                        # Only store a few evidence sentences per relationship
                        if len(relationships[edge_key]["evidence"]) < 3:
                            relationships[edge_key]["evidence"].append(sentence)
        
        # Convert to list format
        result = []
        for (entity1, entity2), data in relationships.items():
            if data["weight"] >= self.min_relationship_strength:
                result.append({
                    "source": entity1,
                    "target": entity2,
                    "weight": data["weight"],
                    "evidence": data["evidence"],
                    "relationship_type": "CO_OCCURRENCE"
                })
        
        return result
    
    def extract_pattern_relationships(
        self,
        sentence_entities: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Extract relationships based on linguistic patterns"""
        relationships = []
        
        for sent_data in sentence_entities:
            sentence = sent_data["sentence"]
            entities = {ent["text"]: ent for ent in sent_data["entities"]}
            
            # Try each pattern
            for pattern, rel_type in self.relationship_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                
                for match in matches:
                    # Check if matched terms are entities
                    source = match.group(1)
                    target = match.group(3) if match.lastindex >= 3 else None
                    
                    if target and (source in entities or target in entities):
                        relationships.append({
                            "source": source,
                            "target": target,
                            "weight": 2.0,  # Pattern-based relationships have higher weight
                            "evidence": [sentence],
                            "relationship_type": rel_type
                        })
        
        return relationships
    
    def merge_relationships(
        self,
        cooccurrence_rels: List[Dict[str, any]],
        pattern_rels: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Merge and deduplicate relationships from different extraction methods"""
        merged = {}
        
        # Add co-occurrence relationships
        for rel in cooccurrence_rels:
            edge_key = tuple(sorted([rel["source"], rel["target"]]))
            if edge_key not in merged:
                merged[edge_key] = rel
        
        # Merge pattern relationships
        for rel in pattern_rels:
            edge_key = tuple(sorted([rel["source"], rel["target"]]))
            if edge_key in merged:
                # Update existing relationship
                merged[edge_key]["weight"] += rel["weight"]
                merged[edge_key]["evidence"].extend(rel["evidence"])
                merged[edge_key]["relationship_type"] = rel["relationship_type"]  # Prefer semantic type
            else:
                merged[edge_key] = rel
        
        # Convert back to list and limit evidence
        result = []
        for (entity1, entity2), data in merged.items():
            # Keep only unique evidence sentences (up to 3)
            unique_evidence = list(dict.fromkeys(data["evidence"]))[:3]
            
            result.append({
                "source": entity1,
                "target": entity2,
                "weight": data["weight"],
                "evidence": unique_evidence,
                "relationship_type": data.get("relationship_type", "CO_OCCURRENCE")
            })
        
        return result
    
    def extract_all_relationships(
        self,
        sentence_entities: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Extract all relationships using multiple methods"""
        cooccurrence_rels = self.extract_cooccurrence_relationships(sentence_entities)
        pattern_rels = self.extract_pattern_relationships(sentence_entities)
        
        return self.merge_relationships(cooccurrence_rels, pattern_rels)

