import networkx as nx
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import community.community_louvain as community_louvain
from app.models.schemas import Node, Edge, GraphData, EntityType, GraphAnalytics


class GraphBuilder:
    """Build and analyze knowledge graphs from extracted entities and relationships"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_graph(
        self,
        entities: Dict[str, Dict[str, any]],
        relationships: List[Dict[str, any]]
    ) -> GraphData:
        """Build a graph from entities and relationships"""
        self.graph.clear()
        
        # Add nodes
        nodes = []
        for entity_key, entity_data in entities.items():
            node_id = entity_data["original_name"]
            self.graph.add_node(
                node_id,
                entity_type=entity_data["type"],
                count=entity_data["count"]
            )
        
        # Add edges
        edges = []
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            weight = rel["weight"]
            evidence = rel["evidence"]
            rel_type = rel.get("relationship_type", "CO_OCCURRENCE")
            
            # Only add edge if both nodes exist
            if self.graph.has_node(source) and self.graph.has_node(target):
                self.graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    evidence=evidence,
                    relationship_type=rel_type
                )
        
        # Convert to output format
        nodes = self._build_nodes()
        edges = self._build_edges()
        
        return GraphData(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "density": self._calculate_density(),
            }
        )
    
    def _build_nodes(self) -> List[Node]:
        """Convert networkx nodes to Node schema"""
        nodes = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            degree = self.graph.degree(node_id)
            
            nodes.append(Node(
                id=node_id,
                group=EntityType(node_data.get("entity_type", "UNKNOWN")),
                value=degree,  # Node size = degree
                metadata={
                    "count": node_data.get("count", 0),
                    "degree": degree
                }
            ))
        
        return nodes
    
    def _build_edges(self) -> List[Edge]:
        """Convert networkx edges to Edge schema"""
        edges = []
        
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]
            evidence = edge_data.get("evidence", [])
            
            # Create title from first evidence sentence
            title = evidence[0] if evidence else f"{source} co-occurs with {target}"
            
            edges.append(Edge(
                source=source,
                target=target,
                value=edge_data.get("weight", 1.0),
                title=title,
                metadata={
                    "all_evidence": evidence,
                    "relationship_type": edge_data.get("relationship_type", "CO_OCCURRENCE")
                }
            ))
        
        return edges
    
    def compute_analytics(self) -> GraphAnalytics:
        """Compute graph analytics and statistics"""
        if len(self.graph.nodes()) == 0:
            return GraphAnalytics(
                total_nodes=0,
                total_edges=0,
                density=0.0,
                avg_degree=0.0,
                communities=[],
                centrality_scores={},
                entity_counts={}
            )
        
        # Basic metrics
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        density = self._calculate_density()
        avg_degree = sum(dict(self.graph.degree()).values()) / total_nodes if total_nodes > 0 else 0
        
        # Community detection
        communities = self._detect_communities()
        
        # Centrality measures
        centrality_scores = self._compute_centrality()
        
        # Entity type counts
        entity_counts = self._count_entity_types()
        
        return GraphAnalytics(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=density,
            avg_degree=avg_degree,
            communities=communities,
            centrality_scores=centrality_scores,
            entity_counts=entity_counts
        )
    
    def _calculate_density(self) -> float:
        """Calculate graph density"""
        return nx.density(self.graph) if len(self.graph.nodes()) > 0 else 0.0
    
    def _detect_communities(self) -> List[List[str]]:
        """Detect communities using Louvain algorithm"""
        if len(self.graph.nodes()) < 2:
            return [[node] for node in self.graph.nodes()]
        
        try:
            partition = community_louvain.best_partition(self.graph)
            
            # Group nodes by community
            communities_dict = defaultdict(list)
            for node, community_id in partition.items():
                communities_dict[community_id].append(node)
            
            return list(communities_dict.values())
        except:
            return [[node] for node in self.graph.nodes()]
    
    def _compute_centrality(self) -> Dict[str, float]:
        """Compute betweenness centrality for all nodes"""
        if len(self.graph.nodes()) < 2:
            return {node: 0.0 for node in self.graph.nodes()}
        
        try:
            centrality = nx.betweenness_centrality(self.graph)
            # Return top 20 by centrality
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_centrality[:20])
        except:
            return {}
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type"""
        counts = defaultdict(int)
        
        for node_id in self.graph.nodes():
            entity_type = self.graph.nodes[node_id].get("entity_type", "UNKNOWN")
            counts[entity_type] += 1
        
        return dict(counts)
    
    def filter_graph(
        self,
        min_degree: int = 1,
        entity_types: List[str] = None,
        top_n: int = None
    ) -> GraphData:
        """Filter graph based on criteria"""
        filtered_graph = self.graph.copy()
        
        # Filter by degree
        nodes_to_remove = [
            node for node in filtered_graph.nodes()
            if filtered_graph.degree(node) < min_degree
        ]
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # Filter by entity type
        if entity_types:
            nodes_to_remove = [
                node for node in filtered_graph.nodes()
                if filtered_graph.nodes[node].get("entity_type") not in entity_types
            ]
            filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # Keep only top N nodes by degree
        if top_n:
            degrees = dict(filtered_graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            nodes_to_keep = [node for node, _ in sorted_nodes[:top_n]]
            nodes_to_remove = [node for node in filtered_graph.nodes() if node not in nodes_to_keep]
            filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # Rebuild graph data
        temp_graph = self.graph
        self.graph = filtered_graph
        result = GraphData(
            nodes=self._build_nodes(),
            edges=self._build_edges(),
            metadata={"filtered": True}
        )
        self.graph = temp_graph
        
        return result
    
    def merge_graphs(self, base_entities: Dict[str, Dict], base_relationships: List[Dict],
                     new_entities: Dict[str, Dict], new_relationships: List[Dict]) -> GraphData:
        """
        Merge two graphs: combine nodes (keeping max counts), aggregate edge weights, merge evidence
        """
        # Merge entities: union, keep highest count
        merged_entities = dict(base_entities)
        for key, entity in new_entities.items():
            if key in merged_entities:
                # Keep existing but update count to max
                merged_entities[key]["count"] = max(
                    merged_entities[key].get("count", 1),
                    entity.get("count", 1)
                )
            else:
                merged_entities[key] = entity
        
        # Merge relationships: aggregate weights, combine evidence
        rel_map = {}
        for rel in base_relationships:
            key = tuple(sorted([rel["source"], rel["target"]]))
            rel_map[key] = {
                "source": rel["source"],
                "target": rel["target"],
                "weight": rel.get("weight", 1.0),
                "evidence": rel.get("evidence", []),
                "relationship_type": rel.get("relationship_type", "CO_OCCURRENCE")
            }
        
        for rel in new_relationships:
            key = tuple(sorted([rel["source"], rel["target"]]))
            if key in rel_map:
                # Aggregate weight and merge evidence
                rel_map[key]["weight"] += rel.get("weight", 1.0)
                existing_evidence = rel_map[key]["evidence"]
                new_evidence = rel.get("evidence", [])
                # Combine and deduplicate evidence (keep top 5)
                combined = list(dict.fromkeys(existing_evidence + new_evidence))
                rel_map[key]["evidence"] = combined[:5]
            else:
                rel_map[key] = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "weight": rel.get("weight", 1.0),
                    "evidence": rel.get("evidence", []),
                    "relationship_type": rel.get("relationship_type", "CO_OCCURRENCE")
                }
        
        merged_relationships = list(rel_map.values())
        
        # Build graph from merged data
        return self.build_graph(merged_entities, merged_relationships)

