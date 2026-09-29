from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime
import uuid
from pathlib import Path

from app.services.pubmed_service import PubMedService
from app.services.ctgov_service import ClinicalTrialsService
from app.services.google_scholar_service import GoogleScholarService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.document_chunker import DocumentChunker
from app.services.ner_service import NERService
from app.services.relationship_extractor import RelationshipExtractor
from app.services.graph_builder import GraphBuilder
from app.services.content_insight_agent import ContentInsightAgent


class AgenticAIService:
    """
    Agentic AI service that can autonomously:
    1. Search for research papers on topics
    2. Download and analyze papers
    3. Build knowledge graphs
    4. Generate insights and hypotheses
    5. Find connections between papers
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.pubmed_service = PubMedService()
        self.google_scholar_service = GoogleScholarService()
        self.ctgov_service = ClinicalTrialsService()
        self.rag_service = RAGService(llm_service=llm_service)
        self.document_chunker = DocumentChunker(chunk_size=500, overlap=100)
        self.ner_service = NERService()
        self.relationship_extractor = RelationshipExtractor()
        self.graph_builder = GraphBuilder()
        
    async def autonomous_research(
        self,
        research_topic: str,
        max_papers: int = 10,
        search_strategy: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform autonomous research on a given topic
        
        Args:
            research_topic: The topic to research
            max_papers: Maximum number of papers to analyze
            search_strategy: "comprehensive", "recent", or "high_impact"
            
        Returns:
            Dict with research results, insights, and recommendations
        """
        print(f"🔬 Starting autonomous research on: {research_topic}")
        
        # Step 1: Generate search queries using LLM
        search_queries = await self._generate_search_queries(research_topic, search_strategy)
        print(f"📝 Generated {len(search_queries)} search queries")
        
        # Step 2: Search for papers
        all_papers = []
        for query in search_queries:
            papers = await self._search_papers(query, max_papers // len(search_queries))
            all_papers.extend(papers)
        
        # Remove duplicates and limit
        unique_papers = self._deduplicate_papers(all_papers)[:max_papers]
        print(f"📚 Found {len(unique_papers)} unique papers")
        
        # Step 3: Analyze papers and build knowledge graph
        analysis_results = await self._analyze_papers(unique_papers, research_topic)
        
        # Step 4: Generate insights and hypotheses
        insights = await self._generate_research_insights(analysis_results, research_topic)
        
        # Step 5: Find research gaps and recommendations
        recommendations = await self._generate_recommendations(analysis_results, research_topic)
        
        return {
            "research_topic": research_topic,
            "papers_analyzed": len(unique_papers),
            "papers": unique_papers,
            "knowledge_graph": analysis_results["graph_data"],
            "insights": insights,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "search_strategy": search_strategy
        }
    
    async def _generate_search_queries(
        self, 
        research_topic: str, 
        strategy: str
    ) -> List[str]:
        """Generate optimized search queries for the research topic"""
        
        strategy_prompts = {
            "comprehensive": "Generate diverse search queries covering different aspects, methodologies, and related fields",
            "recent": "Focus on recent developments and emerging trends",
            "high_impact": "Prioritize high-impact papers, reviews, and landmark studies"
        }
        
        prompt = f"""
        Generate 3-5 optimized PubMed search queries for researching: "{research_topic}"
        
        Strategy: {strategy_prompts.get(strategy, strategy_prompts["comprehensive"])}
        
        Requirements:
        - Use proper PubMed search syntax (MeSH terms, Boolean operators)
        - Include synonyms and related terms
        - Vary the scope (broad to specific)
        - Include recent years filter for recent strategy
        
        Return as JSON array of strings:
        ["query1", "query2", "query3"]
        """
        
        try:
            response = await self.llm_service.chat([
                {"role": "user", "content": prompt}
            ])
            
            queries = json.loads(response)
            return queries if isinstance(queries, list) else [research_topic]
        except Exception as e:
            print(f"Failed to generate search queries: {e}")
            return [research_topic]
    
    async def _search_papers(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search for papers using PubMed and enrich with Google Scholar PDF links"""
        try:
            # Search PubMed
            pmids = self.pubmed_service.search(query, max_results)
            
            # Fetch abstracts for all PMIDs at once
            papers = self.pubmed_service.fetch_abstracts(pmids)
            
            # For papers without PDF links, try Google Scholar
            print(f"🔍 Enriching {len(papers)} papers with Google Scholar PDF links...")
            for paper in papers:
                if not paper.get('pdf_url') and not paper.get('pmc_id'):
                    # Try to find PDF on Google Scholar
                    try:
                        pdf_url = self.google_scholar_service.find_pdf_for_paper(
                            title=paper.get('title', ''),
                            authors=paper.get('authors', [])
                        )
                        if pdf_url:
                            paper['pdf_url'] = pdf_url
                            paper['pdf_source'] = 'google_scholar'
                            print(f"   ✓ Found PDF on Google Scholar: {paper.get('title', '')[:50]}")
                    except Exception as e:
                        print(f"   ⚠️ Google Scholar lookup failed for: {paper.get('title', '')[:50]} - {e}")
            
            return papers
        except Exception as e:
            print(f"Error searching papers for query '{query}': {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on PMID or title"""
        seen_pmids = set()
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            pmid = paper.get("pmid", "")
            title = paper.get("title", "").lower().strip()
            
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_papers.append(paper)
            elif title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    async def _analyze_papers(
        self, 
        papers: List[Dict[str, Any]], 
        research_topic: str
    ) -> Dict[str, Any]:
        """Analyze papers and build knowledge graph"""
        print(f"🔍 Analyzing {len(papers)} papers...")
        
        all_entities = {}
        all_relationships = []
        all_chunks = []
        
        for i, paper in enumerate(papers):
            print(f"📄 Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
            # Extract text content (abstract + title)
            text_content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            
            if not text_content.strip():
                continue
            
            # Process with NER
            entities = self.ner_service.extract_entities(text_content)
            
            # Format entities for get_unique_entities method
            sentence_entities = [{"entities": entities}]
            
            # Get unique entities in the correct format
            unique_entities = self.ner_service.get_unique_entities(sentence_entities)
            
            # Format entities for relationship extraction (needs sentence context)
            sentence_entities_for_relationships = [{"entities": entities, "sentence": text_content}]
            
            # Extract relationships
            relationships = self.relationship_extractor.extract_all_relationships(
                sentence_entities_for_relationships
            )
            
            # Chunk document for RAG
            chunks = self.document_chunker.chunk_document(
                text=text_content,
                doc_id=f"agentic_paper_{paper.get('pmid', 'unknown')}"
            )
            
            # Accumulate results
            all_entities.update(unique_entities)
            all_relationships.extend(relationships)
            all_chunks.extend(chunks)
        
        # Build knowledge graph
        graph_data = self.graph_builder.build_graph(all_entities, all_relationships)
        
        # Index in RAG
        self.rag_service.index_document(
            doc_id=f"agentic_research_{uuid.uuid4()}",
            text_chunks=all_chunks,
            entities=list(all_entities.keys())
        )
        
        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "chunks": all_chunks,
            "graph_data": graph_data,
            "papers_count": len(papers)
        }
    
    async def _generate_research_insights(
        self, 
        analysis_results: Dict[str, Any], 
        research_topic: str
    ) -> List[Dict[str, Any]]:
        """Generate insights from the research analysis"""
        
        # Use ContentInsightAgent for insights
        insight_agent = ContentInsightAgent(
            nx_graph=self.graph_builder.graph,
            documents_data=analysis_results.get("chunks", []),
            original_nodes=list(analysis_results.get("entities", {}).values())
        )
        
        # Generate insights
        insights = insight_agent.generate_insights(
            focus_entity=research_topic,
            max_results=10
        )
        
        return insights
    
    async def _generate_recommendations(
        self, 
        analysis_results: Dict[str, Any], 
        research_topic: str
    ) -> List[Dict[str, Any]]:
        """Generate research recommendations and gaps"""
        
        prompt = f"""
        Based on the analysis of {analysis_results['papers_count']} research papers on "{research_topic}",
        generate research recommendations and identify knowledge gaps.
        
        Available data:
        - {len(analysis_results['entities'])} unique entities
        - {len(analysis_results['relationships'])} relationships
        - Knowledge graph with {len(analysis_results['graph_data'].nodes)} nodes
        
        Provide:
        1. Key research gaps
        2. Recommended future studies
        3. Emerging trends
        4. Methodological suggestions
        
        Return as JSON with this structure:
        {{
            "research_gaps": ["gap1", "gap2", "gap3"],
            "future_studies": ["study1", "study2", "study3"],
            "emerging_trends": ["trend1", "trend2", "trend3"],
            "methodological_suggestions": ["suggestion1", "suggestion2", "suggestion3"]
        }}
        """
        
        try:
            response = await self.llm_service.chat([
                {"role": "user", "content": prompt}
            ])
            
            recommendations = json.loads(response)
            return recommendations if isinstance(recommendations, dict) else {}
        except Exception as e:
            print(f"Failed to generate recommendations: {e}")
            return {}
    
    async def find_related_papers(
        self, 
        current_papers: List[Dict[str, Any]], 
        max_new_papers: int = 5
    ) -> List[Dict[str, Any]]:
        """Find related papers based on current research"""
        
        # Extract key terms from current papers
        key_terms = []
        for paper in current_papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            
            # Use LLM to extract key terms
            prompt = f"""
            Extract 3-5 key research terms from this paper:
            Title: {title}
            Abstract: {abstract}
            
            Return as JSON array: ["term1", "term2", "term3"]
            """
            
            try:
                response = await self.llm_service.chat([
                    {"role": "user", "content": prompt}
                ])
                terms = json.loads(response)
                if isinstance(terms, list):
                    key_terms.extend(terms)
            except:
                pass
        
        # Search for papers using key terms
        related_papers = []
        for term in key_terms[:5]:  # Limit to avoid too many searches
            papers = await self._search_papers(term, max_new_papers // len(key_terms))
            related_papers.extend(papers)
        
        return self._deduplicate_papers(related_papers)[:max_new_papers]
