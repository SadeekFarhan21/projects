from typing import List, Dict, Optional, Any
import asyncio
import json
from app.config import settings

# Only Anthropic import
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMService:
    """
    LLM-powered features using Anthropic Claude
    Direct Anthropic API integration
    """
    
    def __init__(self):
        
        # Use Anthropic API directly
        anthropic_key = "REDACTED-removed-hardcoded-key"
        self.anthropic_client = None
        
        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                self.enabled = True
                print("✅ LLM service enabled with direct Anthropic API")
            except Exception as e:
                print(f"⚠️  Failed to initialize Anthropic client: {e}")
                self.enabled = False
        else:
            print("⚠️  Anthropic library not available")
            self.enabled = False
    
    async def extract_relationships_from_sentence(
        self,
        sentence: str,
        entities: List[str]
    ) -> List[Dict[str, any]]:
        """
        Use LLM to extract relationships from a sentence
        Returns relationships with semantic labels
        """
        if not self.enabled:
            return []
        
        prompt = self._build_extraction_prompt(sentence, entities)
        
        try:
            if self.anthropic_client:
                return await self._extract_with_anthropic(prompt)
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return []
        
        return []
    
    def _build_extraction_prompt(self, sentence: str, entities: List[str]) -> str:
        """Build prompt for relationship extraction"""
        entities_str = ", ".join(entities)
        
        prompt = f"""Given the following biomedical sentence and entities, extract relationships between the entities.

Sentence: {sentence}

Entities: {entities_str}

For each relationship found, provide:
1. Source entity
2. Target entity
3. Relationship type (e.g., INHIBITS, ACTIVATES, TREATS, CAUSES, ASSOCIATES_WITH)
4. Confidence (0.0 to 1.0)

Return as JSON array:
[{{"source": "...", "target": "...", "type": "...", "confidence": 0.9}}]

If no clear relationships exist, return an empty array.
"""
        return prompt
    
    
    async def _extract_with_anthropic(self, prompt: str) -> List[Dict[str, any]]:
        """Extract using direct Anthropic API"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=messages
            )
            
            content = response.content[0].text
            
            # Parse JSON response
            try:
                relationships = json.loads(content)
                if isinstance(relationships, list):
                    return relationships
                elif isinstance(relationships, dict):
                    # Handle case where LLM returns a single relationship as dict
                    if "relationships" in relationships:
                        return relationships["relationships"] if isinstance(relationships["relationships"], list) else [relationships["relationships"]]
                    elif "source" in relationships and "target" in relationships:
                        # Single relationship returned as dict, wrap in list
                        return [relationships]
                    else:
                        return []
                else:
                    return []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            print(f"Anthropic extraction failed: {e}")
            return []
    
    async def classify_relationship(
        self,
        source: str,
        target: str,
        evidence: str
    ) -> Optional[str]:
        """
        Classify the type of relationship between two entities
        Returns relationship type or None
        """
        if not self.enabled:
            return None
        
        prompt = f"""Given the following biomedical entities and evidence, classify the relationship type.

Source Entity: {source}
Target Entity: {target}
Evidence: {evidence}

Choose the most appropriate relationship type:
- INHIBITS
- ACTIVATES
- TREATS
- CAUSES
- ASSOCIATES_WITH
- BINDS_TO
- REGULATES
- EXPRESSED_IN
- UNKNOWN

Return only the relationship type, nothing else.
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Use direct Anthropic API
            if self.anthropic_client:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-haiku-20240307",
                    max_tokens=20,
                    messages=messages
                )
                return response.content[0].text.strip()
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return None
        
        return None
    
    async def generate_insights(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Generate research insights using LLM analysis of extracted content.
        
        Args:
            prompt: Comprehensive prompt with research data and analysis request
            
        Returns:
            List of insight dictionaries with title, description, entities, etc.
        """
        if not self.enabled:
            return []
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Use direct Anthropic API
            if self.anthropic_client:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.3,
                    messages=messages
                )
                response_text = response.content[0].text
            else:
                return []
            
            # Parse JSON response
            try:
                insights = json.loads(response_text)
                if isinstance(insights, list):
                    return insights
                elif isinstance(insights, dict):
                    # Handle case where LLM returns a single insight as dict
                    # Convert to list format
                    if "insights" in insights:
                        return insights["insights"] if isinstance(insights["insights"], list) else [insights["insights"]]
                    elif "hypotheses" in insights:
                        # Handle hypotheses format from LLM
                        return insights["hypotheses"] if isinstance(insights["hypotheses"], list) else [insights["hypotheses"]]
                    elif "research_hypotheses" in insights:
                        # Handle research_hypotheses format from LLM
                        return insights["research_hypotheses"] if isinstance(insights["research_hypotheses"], list) else [insights["research_hypotheses"]]
                    elif "title" in insights or "description" in insights:
                        # Single insight returned as dict, wrap in list
                        return [insights]
                    else:
                        # Get the keys to help debug what format was received
                        keys = list(insights.keys())[:3]  # Show first 3 keys
                        print(f"Unexpected dict format with keys: {keys}")
                        return []
                else:
                    print(f"Unexpected LLM response format: {type(insights)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM insights JSON: {e}")
                print(f"Response: {response_text}")
                return []
                
        except Exception as e:
            print(f"LLM insight generation failed: {e}")
            return []
        
        return []
    
    async def chat(self, messages: List[Dict[str, str]], model: str = "claude-3-5-sonnet-20241022") -> str:
        """
        General chat/completion using Anthropic Claude
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Claude model to use
            
        Returns:
            Response text
        """
        if not self.enabled:
            return ""
        
        try:
            if self.anthropic_client:
                response = await asyncio.to_thread(
                    self.anthropic_client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=messages
                )
                return response.content[0].text
        except Exception as e:
            print(f"LLM chat failed: {e}")
            return ""
        
        return ""
