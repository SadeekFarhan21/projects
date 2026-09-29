"""
Unified LLM clients for OpenAI and Gemini.

Provides simple wrappers for GPT-5.2 and Gemini 3 Pro.
"""

from typing import Optional


class OpenAIClient:
    """
    OpenAI client wrapper for GPT-5.2 with web search and reasoning support.
    """

    def __init__(self, api_key: str, model: str = "gpt-5.2"):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5.2)
        """
        from openai import OpenAI

        self.api_key = api_key
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        use_web_search: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            use_web_search: Whether to enable web search tool
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            The model's response text
        """
        # Build request parameters
        params = {
            "model": self.model,
            "input": prompt,
        }

        # Add tools if web search enabled
        if use_web_search:
            params["tools"] = [{"type": "web_search"}]

        # Add reasoning config if specified
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        response = self._client.responses.create(**params)
        return response.output_text


class GeminiClient:
    """
    Gemini client wrapper for Gemini 3 Pro.
    """

    def __init__(self, api_key: str, model: str = "gemini-3-pro"):
        """
        Initialize the Gemini client.

        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-3-pro)
        """
        from google import genai
        from google.genai import types

        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_output_tokens: int = 30000,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate

        Returns:
            The model's response text
        """
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        return response.text
