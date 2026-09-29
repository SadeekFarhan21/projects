from typing import List, Dict, Any, Optional
import networkx as nx
from collections import defaultdict, Counter
import json


class ContentInsightAgent:
    """
    Generate meaningful insights by extracting key content from PDFs and graph relationships,
    then feeding this information to LLMs for analysis and hypothesis generation.
    """
    
    def __init__(self, nx_graph: nx.Graph, documents_data: List[Dict] = None, original_nodes: List[Dict] = None):
        self.graph = nx_graph
        self.documents_data = documents_data or []
        self.original_nodes = original_nodes or []
        
        # Extract key information from the graph and documents
        self.entity_summary = self._extract_entity_summary()
        self.relationship_summary = self._extract_relationship_summary()
        self.document_summary = self._extract_document_summary()
        self.key_findings = self._extract_key_findings()
    
    def _extract_entity_summary(self) -> Dict[str, Any]:
        """Extract summary of key entities and their properties"""
        if not self.graph.nodes():
            return {}
        
        print(f"DEBUG: Extracting entity summary from {len(self.graph.nodes())} nodes")
        
        # Get top entities by degree (most connected)
        node_degrees = dict(self.graph.degree())
        print(f"DEBUG: Node degrees calculated: {len(node_degrees)}")
        top_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Group by entity type using original node data
        entity_types = defaultdict(list)
        node_metadata = {node.get('id'): node for node in self.original_nodes if node.get('id')}
        
        print(f"DEBUG: Processing {len(self.graph.nodes())} nodes")
        for node_id in self.graph.nodes():
            try:
                node_data = node_metadata.get(node_id, {})
                entity_type = node_data.get('group', 'UNKNOWN')
                entity_types[entity_type].append({
                    'name': node_id,
                    'degree': node_degrees.get(node_id, 0),
                    'connections': node_degrees.get(node_id, 0),
                    'count': node_data.get('metadata', {}).get('count', 1)
                })
            except Exception as e:
                print(f"DEBUG: Error processing node {node_id}: {e}")
                print(f"DEBUG: Node data: {node_data}")
                raise
        
        # Get entity type distribution
        entity_distribution = Counter()
        for node_data in self.original_nodes:
            if node_data.get('id'):
                entity_distribution[node_data.get('group', 'UNKNOWN')] += 1
        
        return {
            'top_entities': top_entities[:10],
            'entity_types': dict(entity_types),
            'total_entities': len(self.graph.nodes()),
            'entity_distribution': dict(entity_distribution)
        }
    
    def _extract_relationship_summary(self) -> Dict[str, Any]:
        """Extract summary of key relationships"""
        if not self.graph.edges():
            return {}
        
        print(f"DEBUG: Extracting relationship summary from {len(self.graph.edges())} edges")
        
        # Get relationship types
        relationship_types = defaultdict(int)
        edge_weights = []
        
        for source, target, data in self.graph.edges(data=True):
            try:
                rel_type = data.get('relationship_type', 'CO_OCCURRENCE')
                weight = data.get('weight', 1.0)
                relationship_types[rel_type] += 1
                edge_weights.append(weight)
            except Exception as e:
                print(f"DEBUG: Error processing edge {source}->{target}: {e}")
                print(f"DEBUG: Edge data: {data}")
                raise
        
        # Get strongest relationships
        edges_with_weights = [(source, target, data.get('weight', 1.0), data.get('relationship_type', 'CO_OCCURRENCE'), data.get('evidence', ''))
                             for source, target, data in self.graph.edges(data=True)]
        strongest_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)[:10]
        
        return {
            'relationship_types': dict(relationship_types),
            'strongest_relationships': strongest_edges,
            'total_relationships': len(self.graph.edges()),
            'avg_weight': sum(edge_weights) / len(edge_weights) if edge_weights else 0
        }
    
    def _extract_document_summary(self) -> Dict[str, Any]:
        """Extract summary information from documents"""
        if not self.documents_data:
            return {}
        
        total_docs = len(self.documents_data)
        doc_types = Counter([doc.get('type', 'UNKNOWN') for doc in self.documents_data])
        
        # Extract key topics/themes from document titles/names
        doc_names = [doc.get('name', '') for doc in self.documents_data if doc.get('name')]
        
        return {
            'total_documents': total_docs,
            'document_types': dict(doc_types),
            'document_names': doc_names,
            'document_metadata': self.documents_data
        }
    
    def _extract_key_findings(self) -> List[Dict[str, Any]]:
        """Extract key findings and patterns from the data"""
        findings = []
        
        # Finding 1: Most connected entities
        if self.entity_summary.get('top_entities'):
            top_entity = self.entity_summary['top_entities'][0]
            findings.append({
                'type': 'central_entity',
                'title': f"Most Connected Entity: {top_entity[0]}",
                'description': f"Entity '{top_entity[0]}' has {top_entity[1]} connections, making it central to the research",
                'entities': [top_entity[0]],
                'confidence': 0.9
            })
        
        # Finding 2: Strongest relationships
        if self.relationship_summary.get('strongest_relationships'):
            strongest = self.relationship_summary['strongest_relationships'][0]
            findings.append({
                'type': 'strong_relationship',
                'title': f"Strongest Relationship: {strongest[0]} → {strongest[1]}",
                'description': f"Strong {strongest[3]} relationship between {strongest[0]} and {strongest[1]} (weight: {strongest[2]:.2f})",
                'entities': [strongest[0], strongest[1]],
                'evidence': [strongest[4]] if len(strongest) > 4 and strongest[4] else [],
                'confidence': 0.8
            })
        
        # Finding 3: Entity type distribution
        if self.entity_summary.get('entity_distribution'):
            entity_dist = self.entity_summary['entity_distribution']
            most_common_type = max(entity_dist.items(), key=lambda x: x[1])
            findings.append({
                'type': 'entity_dominance',
                'title': f"Dominant Entity Type: {most_common_type[0]}",
                'description': f"{most_common_type[0]} entities dominate the research ({most_common_type[1]} entities)",
                'entities': list(entity_dist.keys()),
                'confidence': 0.7
            })
        
        return findings
    
    def generate_insight_prompt(self, focus_entity: Optional[str] = None) -> str:
        """Generate a comprehensive prompt for LLM analysis"""
        
        prompt = f"""You are a biomedical research analyst. Analyze the following research data and generate meaningful insights and hypotheses.

RESEARCH DATA SUMMARY:
- Total Entities: {self.entity_summary.get('total_entities', 0)}
- Total Relationships: {self.relationship_summary.get('total_relationships', 0)}
- Documents Analyzed: {self.document_summary.get('total_documents', 0)}

KEY ENTITIES:
{json.dumps(self.entity_summary.get('top_entities', [])[:10], indent=2)}

ENTITY TYPE DISTRIBUTION:
{json.dumps(self.entity_summary.get('entity_distribution', {}), indent=2)}

RELATIONSHIP TYPES:
{json.dumps(self.relationship_summary.get('relationship_types', {}), indent=2)}

STRONGEST RELATIONSHIPS:
{json.dumps(self.relationship_summary.get('strongest_relationships', [])[:5], indent=2)}

DOCUMENTS:
{json.dumps(self.document_summary.get('document_names', []), indent=2)}

KEY FINDINGS:
{json.dumps(self.key_findings, indent=2)}"""

        if focus_entity:
            prompt += f"\n\nFOCUS ENTITY: {focus_entity}\nPlease pay special attention to insights related to {focus_entity}."

        prompt += """

TASK: Generate 5-8 meaningful research insights and hypotheses based on this data. For each insight:

1. **Title**: Clear, specific title
2. **Type**: One of: hypothesis, finding, pattern, connection, implication
3. **Description**: Detailed explanation of the insight
4. **Entities**: List of key entities involved
5. **Evidence**: Specific evidence from the data supporting this insight
6. **Confidence**: 0.0-1.0 confidence score
7. **Implications**: What this means for research or clinical practice

Focus on:
- Novel connections between entities
- Patterns that suggest causal relationships
- Clinical or therapeutic implications
- Research gaps or opportunities
- Biomarker or diagnostic potential

Return as JSON array of insight objects."""

        return prompt
    
    def generate_insights(self, focus_entity: Optional[str] = None, max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Generate insights by creating a comprehensive prompt for LLM analysis.
        This method prepares all the data for LLM processing.
        """
        
        # For now, return the key findings we extracted
        # In a full implementation, this would call the LLM service
        insights = []
        
        # Add our extracted findings
        insights.extend(self.key_findings)
        
        # Add some pattern-based insights
        if self.entity_summary.get('top_entities'):
            top_entities = self.entity_summary['top_entities'][:3]
            if len(top_entities) >= 2:
                insights.append({
                    'type': 'pattern',
                    'title': f"Research Focus: {top_entities[0][0]} and {top_entities[1][0]}",
                    'description': f"The research appears to focus on the relationship between {top_entities[0][0]} and {top_entities[1][0]}, both highly connected entities",
                    'entities': [top_entities[0][0], top_entities[1][0]],
                    'confidence': 0.8
                })
        
        # Add relationship pattern insights
        if self.relationship_summary.get('relationship_types'):
            rel_types = self.relationship_summary['relationship_types']
            if len(rel_types) > 1:
                most_common_rel = max(rel_types.items(), key=lambda x: x[1])
                insights.append({
                    'type': 'pattern',
                    'title': f"Dominant Relationship Type: {most_common_rel[0]}",
                    'description': f"The research primarily focuses on {most_common_rel[0]} relationships ({most_common_rel[1]} instances), suggesting this is a key mechanism",
                    'entities': [],
                    'confidence': 0.7
                })
        
        return insights[:max_results]
    
    def get_llm_prompt(self, focus_entity: Optional[str] = None) -> str:
        """Get the formatted prompt for LLM analysis"""
        return self.generate_insight_prompt(focus_entity)
