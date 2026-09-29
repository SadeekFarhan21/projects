"""
Agent 1: Question Transformer

Transforms a natural text medical question into a causal question
formulated for Target Trial Emulation (TTE).

Uses GPT-5.2 Thinking with web search to ensure the question is
properly framed for causal inference.
"""

from ..utils.llm_client import OpenAIClient

QUESTION_TRANSFORM_PROMPT = """You are an expert epidemiologist specializing in causal inference and Target Trial Emulation (TTE).

## Your Task

Transform the following natural-language medical question into a formal causal question suitable for Target Trial Emulation.

**Input Question:**
{question}

## What You Need To Do

1. **Search for clinical context** using web search:
   - Current clinical guidelines
   - Relevant trials or studies
   - Standard definitions

2. **Extract PICO components:**
   - **P (Population)**: Who?
   - **I (Intervention)**: What treatment/exposure?
   - **C (Comparator)**: What's the comparison? (Active comparator preferred)
   - **O (Outcome)**: What outcome and in what time window?

3. **Frame as causal question:**
   "What is the effect of [I] compared to [C] on [O] among [P]?"

4. **Define time zero:**
   When does follow-up start? This must coincide with eligibility AND treatment assignment.

## Output Format

### Causal Question
[Single sentence in PICO format]

### PICO Components

| Component | Description | Data Operationalization |
|-----------|-------------|------------------------|
| **Population** | | |
| **Intervention** | | |
| **Comparator** | | |
| **Outcome** | | |

### Time Zero
[How is time zero defined? Why does this avoid immortal time bias?]

### Clinical Context
[Summary from web search - what is known about this topic]

### Assumptions Made
[Any assumptions to resolve ambiguities]

### Feasibility Notes
[Can this be answered with observational data? Any concerns?]
"""


class QuestionAgent:
    """
    Agent that transforms natural text medical questions into causal questions for TTE.

    Uses GPT-5.2 Thinking with web search for clinical context.
    """

    def __init__(self, client: OpenAIClient):
        """
        Initialize the Question Agent.

        Args:
            client: OpenAI client for LLM calls
        """
        self.client = client

    def transform(self, question: str) -> str:
        """
        Transform a medical question into a causal question for TTE.

        Args:
            question: The natural text medical question

        Returns:
            The transformed causal question with PICO components
        """
        prompt = QUESTION_TRANSFORM_PROMPT.format(question=question)
        response = self.client.generate(
            prompt,
            use_web_search=True,
            reasoning_effort="high",
        )
        return response
