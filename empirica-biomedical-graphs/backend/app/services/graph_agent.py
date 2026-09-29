from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import json


class GraphConversationalAgent:
    """
    Conversational agent to reason over a biomedical knowledge graph.
    Provides tool-like methods for querying the graph that can be orchestrated
    by an external LLM service.
    """

    def __init__(self, nx_graph: nx.Graph, llm_service=None):
        self.graph = nx_graph
        self.llm_service = llm_service
        self.conversation_history = []

    def get_neighbors(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        if entity not in self.graph:
            return {"entity": entity, "neighbors": []}
        
        visited = {entity}
        frontier = {entity}
        layers = []
        for _ in range(max(1, depth)):
            next_frontier = set()
            layer = []
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    edge = self.graph.edges[node, neighbor]
                    layer.append({
                        "source": node,
                        "target": neighbor,
                        "weight": edge.get("weight", 1.0),
                        "relationship_type": edge.get("relationship_type", "CO_OCCURRENCE"),
                        "evidence": edge.get("evidence", [])[:3],
                    })
            layers.append(layer)
            frontier = next_frontier
            if not frontier:
                break
        
        return {"entity": entity, "layers": layers}

    def shortest_path(self, source: str, target: str, k_paths: int = 1) -> Dict[str, Any]:
        """Find shortest path using Dijkstra's algorithm with edge weights"""
        if source not in self.graph or target not in self.graph:
            return {"paths": []}
        try:
            # Use Dijkstra's algorithm: shortest path by weight (not hop count)
            # NetworkX uses Dijkstra by default for weighted graphs
            path = nx.shortest_path(self.graph, source=source, target=target, weight='weight')
            paths = [path]  # Just return the single shortest weighted path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"paths": []}
        
        detailed = []
        for path in paths:
            edges = []
            total_weight = 0
            for a, b in zip(path, path[1:]):
                data = self.graph.edges[a, b]
                weight = data.get("weight", 1.0)
                total_weight += weight
                edges.append({
                    "source": a,
                    "target": b,
                    "weight": weight,
                    "relationship_type": data.get("relationship_type", "CO_OCCURRENCE"),
                    "evidence": data.get("evidence", [])[:3],
                })
            detailed.append({
                "nodes": path, 
                "edges": edges,
                "total_weight": total_weight
            })
        return {"paths": detailed}

    def common_connections(self, entities: List[str], min_degree: int = 1) -> Dict[str, Any]:
        present = [e for e in entities if e in self.graph]
        if len(present) < 2:
            return {"common": []}
        neighbor_sets = [set(self.graph.neighbors(e)) for e in present]
        common = set.intersection(*neighbor_sets) if neighbor_sets else set()
        result = []
        for n in common:
            deg = self.graph.degree(n)
            if deg >= min_degree:
                result.append({"entity": n, "degree": deg})
        result.sort(key=lambda x: x["degree"], reverse=True)
        return {"common": result}

    def subgraph(self, center_entities: List[str], depth: int = 1) -> Dict[str, Any]:
        nodes_to_include = set(center_entities)
        for e in center_entities:
            if e not in self.graph:
                continue
            for _ in range(depth):
                neighbors = set()
                for n in nodes_to_include.copy():
                    if n in self.graph:
                        neighbors.update(self.graph.neighbors(n))
                nodes_to_include.update(neighbors)
        sg = self.graph.subgraph(nodes_to_include).copy()
        nodes = list(sg.nodes())
        edges = [[u, v] for u, v in sg.edges()]
        return {"nodes": nodes, "edges": edges}
    
    async def chat(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Conversational interface for graph queries
        Uses pattern matching first, falls back to LLM for complex queries
        """
        # Try pattern matching first (fast, no cost)
        pattern_result = self._try_pattern_match(user_message)
        if pattern_result:
            return pattern_result
        
        # If pattern matching failed and LLM is available, use it
        if not self.llm_service or not self.llm_service.enabled:
            # No LLM and pattern matching failed
            return self._pattern_match_chat(user_message)
        
        # Prepare graph context
        graph_stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "sample_entities": list(self.graph.nodes())[:20]
        }
        
        # Build system prompt with available tools
        system_prompt = f"""You are a biomedical knowledge graph assistant. You help users explore relationships between biomedical entities.

Available Graph Tools:
1. get_neighbors(entity, depth) - Find neighboring entities
2. shortest_path(source, target) - Find paths between entities
3. common_connections(entities) - Find shared connections
4. subgraph(entities, depth) - Extract a subgraph

Current Graph: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges

When a user asks about the graph, determine which tool(s) to use and extract the relevant entity names from their question. Match entity names to those in the graph (case-insensitive).

Respond in JSON format:
{{
  "tool": "tool_name",
  "params": {{"param": "value"}},
  "entities": ["entity1", "entity2"],
  "explanation": "Natural language explanation of what you'll do"
}}

If no tool is needed, return:
{{
  "tool": null,
  "response": "Your direct answer"
}}"""

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({
            "role": "user",
            "content": f"Graph sample entities: {', '.join(graph_stats['sample_entities'][:10])}...\n\nUser question: {user_message}"
        })
        
        try:
            # Use direct Anthropic API
            if self.llm_service.anthropic_client:
                import asyncio
                response = await asyncio.to_thread(
                    self.llm_service.anthropic_client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    messages=messages
                )
                content = response.content[0].text
            else:
                return self._pattern_match_chat(user_message)
            
            # Parse LLM response
            llm_result = json.loads(content)
            
            # If LLM wants to use a tool, execute it
            if llm_result.get("tool"):
                tool_name = llm_result["tool"]
                params = llm_result.get("params", {})
                
                # Match entities in the graph
                entities_to_match = llm_result.get("entities", [])
                matched_entities = self._match_entities(entities_to_match)
                
                # Execute the tool
                if tool_name == "get_neighbors":
                    entity = matched_entities[0] if matched_entities else params.get("entity", "")
                    result = self.get_neighbors(entity, depth=params.get("depth", 1))
                elif tool_name == "shortest_path":
                    source = matched_entities[0] if len(matched_entities) > 0 else params.get("source", "")
                    target = matched_entities[1] if len(matched_entities) > 1 else params.get("target", "")
                    result = self.shortest_path(source, target)
                elif tool_name == "common_connections":
                    entities = matched_entities if matched_entities else params.get("entities", [])
                    result = self.common_connections(entities)
                elif tool_name == "subgraph":
                    entities = matched_entities if matched_entities else params.get("entities", [])
                    result = self.subgraph(entities, depth=params.get("depth", 1))
                else:
                    result = {"error": "Unknown tool"}
                
                # Format result for user
                return self._format_tool_result(tool_name, result, llm_result.get("explanation", ""))
            else:
                # Direct response from LLM
                return {
                    "answer": llm_result.get("response", "I'm not sure how to help with that."),
                    "tool_calls": [],
                    "relevant_nodes": [],
                    "relevant_edges": [],
                    "citations": []
                }
                
        except Exception as e:
            return self._pattern_match_chat(user_message)
    
    def _match_entities(self, query_entities: List[str]) -> List[str]:
        """Match query entities to actual graph nodes with fuzzy matching"""
        matched = []
        graph_nodes = list(self.graph.nodes())
        
        for query_entity in query_entities:
            query_lower = query_entity.lower().strip()
            best_match = None
            best_score = 0
            
            # Try exact match first
            for node in graph_nodes:
                if node.lower() == query_lower:
                    matched.append(node)
                    break
            else:
                # Try substring match
                for node in graph_nodes:
                    node_lower = node.lower()
                    if query_lower in node_lower:
                        score = len(query_lower) / len(node_lower)
                        if score > best_score:
                            best_score = score
                            best_match = node
                    elif node_lower in query_lower:
                        score = len(node_lower) / len(query_lower)
                        if score > best_score:
                            best_score = score
                            best_match = node
                
                # Try fuzzy matching by checking if words overlap
                if not best_match:
                    query_words = set(query_lower.split())
                    for node in graph_nodes:
                        node_words = set(node.lower().split())
                        overlap = query_words & node_words
                        if overlap:
                            score = len(overlap) / max(len(query_words), len(node_words))
                            if score > best_score:
                                best_score = score
                                best_match = node
                
                if best_match and best_score > 0.3:  # At least 30% similarity
                    matched.append(best_match)
        
        return matched
    
    def _find_similar_entities(self, query: str, limit: int = 5) -> List[str]:
        """Find similar entities for suggestions"""
        query_lower = query.lower().strip()
        graph_nodes = list(self.graph.nodes())
        suggestions = []
        
        for node in graph_nodes:
            node_lower = node.lower()
            # Check if any words match
            if any(word in node_lower for word in query_lower.split()):
                suggestions.append(node)
            # Check if starts with same letter
            elif query_lower and node_lower.startswith(query_lower[0]):
                suggestions.append(node)
        
        return suggestions[:limit]
    
    def _format_tool_result(self, tool_name: str, result: Dict, explanation: str) -> Dict[str, Any]:
        """Format tool execution result for chat response"""
        response = {
            "answer": explanation,
            "tool_calls": [tool_name],
            "relevant_nodes": [],
            "relevant_edges": [],
            "citations": []
        }
        
        if tool_name == "get_neighbors":
            layers = result.get("layers", [])
            if layers:
                neighbors = list({item["target"] for item in layers[0]})
                response["answer"] = f"{explanation}\n\nNeighbors: {', '.join(neighbors[:15])}"
                response["relevant_nodes"] = [result["entity"]] + neighbors
                response["relevant_edges"] = [[result["entity"], t] for t in neighbors]
                response["citations"] = [e for item in layers[0] for e in item.get("evidence", [])][:3]
        
        elif tool_name == "shortest_path":
            paths = result.get("paths", [])
            if paths:
                path = paths[0]
                nodes = path["nodes"]
                edges = path["edges"]
                response["answer"] = f"{explanation}\n\nPath: {' → '.join(nodes)}"
                response["relevant_nodes"] = nodes
                response["relevant_edges"] = [[e["source"], e["target"]] for e in edges]
                response["citations"] = [evi for e in edges for evi in e.get("evidence", [])][:3]
            else:
                response["answer"] = f"{explanation}\n\nNo path found."
        
        elif tool_name == "common_connections":
            commons = result.get("common", [])
            if commons:
                response["answer"] = f"{explanation}\n\nCommon: {', '.join([c['entity'] for c in commons[:15]])}"
                response["relevant_nodes"] = [c["entity"] for c in commons]
        
        return response
    
    def _try_pattern_match(self, message: str) -> Optional[Dict[str, Any]]:
        """Try to answer using pattern matching. Returns None if it can't handle the query."""
        text = message.lower().strip()
        
        # Only handle queries that clearly match our patterns
        # Stats queries
        if "how many" in text and ("nodes" in text or "entities" in text):
            return {
                "answer": f"The graph has {self.graph.number_of_nodes()} nodes.",
                "tool_calls": [],
                "relevant_nodes": [],
                "relevant_edges": [],
                "citations": []
            }
        elif "how many" in text and ("edges" in text or "connections" in text or "relationships" in text):
            return {
                "answer": f"The graph has {self.graph.number_of_edges()} edges.",
                "tool_calls": [],
                "relevant_nodes": [],
                "relevant_edges": [],
                "citations": []
            }
        
        # Neighbor queries
        if "neighbor" in text:
            for pattern in ["neighbors of ", "neighbor of "]:
                if pattern in text:
                    entity_query = text.split(pattern, 1)[1].strip().rstrip("?.,!")
                    matched = self._match_entities([entity_query])
                    if matched:
                        result = self.get_neighbors(matched[0], depth=1)
                        layers = result.get("layers", [])
                        if layers:
                            neighbors = list({item["target"] for item in layers[0]})
                            match_note = f" (matched '{entity_query}' to '{matched[0]}')" if matched[0].lower() != entity_query.lower() else ""
                            return {
                                "answer": f"Neighbors of {result['entity']}{match_note}:\n{', '.join(neighbors[:20])}",
                                "tool_calls": ["get_neighbors"],
                                "relevant_nodes": [result["entity"]] + neighbors[:20],
                                "relevant_edges": [[result["entity"], t] for t in neighbors[:20]],
                                "citations": [e for item in layers[0] for e in item.get("evidence", [])][:3]
                            }
                        else:
                            return {
                                "answer": f"'{matched[0]}' has no neighbors in the graph.",
                                "tool_calls": [],
                                "relevant_nodes": [],
                                "relevant_edges": [],
                                "citations": []
                            }
                    else:
                        # No match found - provide suggestions
                        suggestions = self._find_similar_entities(entity_query, limit=5)
                        if suggestions:
                            return {
                                "answer": f"❌ Couldn't find '{entity_query}' in the graph.\n\n💡 Did you mean one of these?\n• " + "\n• ".join(suggestions),
                                "tool_calls": [],
                                "relevant_nodes": suggestions,
                                "relevant_edges": [],
                                "citations": []
                            }
                        else:
                            return {
                                "answer": f"❌ Entity '{entity_query}' not found in the graph.\n\n📊 The graph has {self.graph.number_of_nodes()} nodes. Try a different entity name.",
                                "tool_calls": [],
                                "relevant_nodes": [],
                                "relevant_edges": [],
                                "citations": []
                            }
        
        # Path queries
        if ("path" in text or "connect" in text) and " between " in text and " and " in text:
            try:
                between_part = text.split(" between ", 1)[1]
                parts = between_part.split(" and ", 1)
                entity_a = parts[0].strip().rstrip("?.,!")
                entity_b = parts[1].strip().rstrip("?.,!")
                
                matched = self._match_entities([entity_a, entity_b])
                if len(matched) >= 2:
                    result = self.shortest_path(matched[0], matched[1])
                    paths = result.get("paths", [])
                    if paths:
                        path = paths[0]
                        nodes = path["nodes"]
                        edges = path["edges"]
                        total_weight = path.get("total_weight", 0)
                        return {
                            "answer": f"Shortest path from {matched[0]} to {matched[1]} (weight: {total_weight:.1f}):\n{' → '.join(nodes)}",
                            "tool_calls": ["shortest_path"],
                            "relevant_nodes": nodes,
                            "relevant_edges": [[e["source"], e["target"]] for e in edges],
                            "citations": [evi for e in edges for evi in e.get("evidence", [])][:3]
                        }
                    else:
                        return {
                            "answer": f"No path found between {matched[0]} and {matched[1]}.",
                            "tool_calls": [],
                            "relevant_nodes": [],
                            "relevant_edges": [],
                            "citations": []
                        }
            except:
                pass
        
        # Common connections queries
        if "common" in text or "connects" in text:
            entities_text = text
            for prefix in ["common connections ", "what connects ", "connects "]:
                if prefix in text:
                    entities_text = text.split(prefix, 1)[1].strip().rstrip("?.,!")
                    break
            
            import re
            entity_names = re.split(r',|\s+and\s+', entities_text)
            entity_names = [e.strip() for e in entity_names if e.strip()]
            
            if len(entity_names) >= 2:
                matched = self._match_entities(entity_names)
                if len(matched) >= 2:
                    result = self.common_connections(matched)
                    commons = result.get("common", [])
                    if commons:
                        return {
                            "answer": f"Common connections for {', '.join(matched)}: {', '.join([c['entity'] for c in commons[:15]])}",
                            "tool_calls": ["common_connections"],
                            "relevant_nodes": matched + [c["entity"] for c in commons[:15]],
                            "relevant_edges": [],
                            "citations": []
                        }
        
        # Return None if pattern matching can't handle it
        return None
    
    def _pattern_match_chat(self, message: str) -> Dict[str, Any]:
        """Fallback help message when no pattern matches and no LLM"""
        # Provide helpful guidance
        return {
            "answer": f"I can help you explore the graph with {self.graph.number_of_nodes()} nodes! Try:\n- 'What are the neighbors of X?'\n- 'Find path between A and B'\n- 'What connects A, B, and C?'\n- 'How many nodes are there?'",
            "tool_calls": [],
            "relevant_nodes": [],
            "relevant_edges": [],
            "citations": []
        }
    
    def _old_pattern_match_chat(self, message: str) -> Dict[str, Any]:
        """OLD: Fallback for when neither pattern matching nor LLM can help"""
        text = message.lower().strip()
        
        print(f"\n🤖 DEBUG: Pattern matching fallback")
        print(f"   User message: '{message}'")
        print(f"   Lowercased text: '{text}'")
        
        # Extract entity names from common patterns
        graph_nodes = list(self.graph.nodes())
        
        # Pattern: "neighbors of X" or "what are neighbors of X"
        if "neighbor" in text:
            print(f"   ✅ Detected 'neighbor' pattern")
            # Try to extract entity name
            for pattern in ["neighbors of ", "neighbor of "]:
                if pattern in text:
                    entity_query = text.split(pattern, 1)[1].strip().rstrip("?.,!")
                    print(f"   📝 Extracted entity query: '{entity_query}'")
                    # Match to actual node with fuzzy matching
                    matched = self._match_entities([entity_query])
                    if matched:
                        result = self.get_neighbors(matched[0], depth=1)
                        layers = result.get("layers", [])
                        if layers:
                            neighbors = list({item["target"] for item in layers[0]})
                            match_note = f" (matched '{entity_query}' to '{matched[0]}')" if matched[0].lower() != entity_query.lower() else ""
                            return {
                                "answer": f"Neighbors of {result['entity']}{match_note}:\n{', '.join(neighbors[:20])}",
                                "tool_calls": ["get_neighbors"],
                                "relevant_nodes": [result["entity"]] + neighbors[:20],
                                "relevant_edges": [[result["entity"], t] for t in neighbors[:20]],
                                "citations": [e for item in layers[0] for e in item.get("evidence", [])][:3]
                            }
                        else:
                            return {
                                "answer": f"'{matched[0]}' has no neighbors in the graph.",
                                "tool_calls": [],
                                "relevant_nodes": [],
                                "relevant_edges": [],
                                "citations": []
                            }
                    else:
                        # No match found - provide suggestions
                        suggestions = self._find_similar_entities(entity_query, limit=5)
                        if suggestions:
                            return {
                                "answer": f"❌ Couldn't find '{entity_query}' in the graph.\n\n💡 Did you mean one of these?\n• " + "\n• ".join(suggestions),
                                "tool_calls": [],
                                "relevant_nodes": suggestions,
                                "relevant_edges": [],
                                "citations": []
                            }
                        else:
                            return {
                                "answer": f"❌ Entity '{entity_query}' not found in the graph.\n\n📊 The graph has {self.graph.number_of_nodes()} nodes. Try a different entity name.",
                                "tool_calls": [],
                                "relevant_nodes": [],
                                "relevant_edges": [],
                                "citations": []
                            }
        
        # Pattern: "path between A and B" or "find path between A and B"
        if ("path" in text or "connect" in text) and " between " in text and " and " in text:
            try:
                # Extract entities
                between_part = text.split(" between ", 1)[1]
                parts = between_part.split(" and ", 1)
                entity_a = parts[0].strip().rstrip("?.,!")
                entity_b = parts[1].strip().rstrip("?.,!")
                
                matched = self._match_entities([entity_a, entity_b])
                if len(matched) >= 2:
                    result = self.shortest_path(matched[0], matched[1])
                    paths = result.get("paths", [])
                    if paths:
                        path = paths[0]
                        nodes = path["nodes"]
                        edges = path["edges"]
                        return {
                            "answer": f"Shortest path from {matched[0]} to {matched[1]}: {' → '.join(nodes)}",
                            "tool_calls": ["shortest_path"],
                            "relevant_nodes": nodes,
                            "relevant_edges": [[e["source"], e["target"]] for e in edges],
                            "citations": [evi for e in edges for evi in e.get("evidence", [])][:3]
                        }
                    else:
                        return {
                            "answer": f"No path found between {matched[0]} and {matched[1]}.",
                            "tool_calls": [],
                            "relevant_nodes": [],
                            "relevant_edges": [],
                            "citations": []
                        }
            except:
                pass
        
        # Pattern: "common connections" or "what connects A, B, C"
        if "common" in text or "connects" in text:
            # Try to find comma-separated entities or "and" separated
            entities_text = text
            for prefix in ["common connections ", "what connects ", "connects "]:
                if prefix in text:
                    entities_text = text.split(prefix, 1)[1].strip().rstrip("?.,!")
                    break
            
            # Split by comma or "and"
            import re
            entity_names = re.split(r',|\s+and\s+', entities_text)
            entity_names = [e.strip() for e in entity_names if e.strip()]
            
            if len(entity_names) >= 2:
                matched = self._match_entities(entity_names)
                if len(matched) >= 2:
                    result = self.common_connections(matched)
                    commons = result.get("common", [])
                    if commons:
                        return {
                            "answer": f"Common connections for {', '.join(matched)}: {', '.join([c['entity'] for c in commons[:15]])}",
                            "tool_calls": ["common_connections"],
                            "relevant_nodes": matched + [c["entity"] for c in commons[:15]],
                            "relevant_edges": [],
                            "citations": []
                        }
        
        # Stats queries
        if "how many" in text and "nodes" in text:
            return {
                "answer": f"The graph has {self.graph.number_of_nodes()} nodes.",
                "tool_calls": [],
                "relevant_nodes": [],
                "relevant_edges": [],
                "citations": []
            }
        elif "how many" in text and "edges" in text:
            return {
                "answer": f"The graph has {self.graph.number_of_edges()} edges.",
                "tool_calls": [],
                "relevant_nodes": [],
                "relevant_edges": [],
                "citations": []
            }
        
        # Default help
        return {
            "answer": f"I can help you explore the graph with {self.graph.number_of_nodes()} nodes! Try:\n- 'What are the neighbors of X?'\n- 'Find path between A and B'\n- 'What connects A, B, and C?'\n\n💡 Enable LLM for natural language understanding!",
            "tool_calls": [],
            "relevant_nodes": [],
            "relevant_edges": [],
            "citations": []
        }


