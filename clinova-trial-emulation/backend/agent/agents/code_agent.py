"""
Agent 3: Code Generator

Generates Python code for Target Trial Emulation from a validated design spec,
using the SOTAstack methodology as implementation reference and resolved OMOP concept IDs.
"""

from ..utils.llm_client import GeminiClient

CODE_GENERATION_PROMPT = """You are an expert Python developer specializing in causal inference and Target Trial Emulation.

## Your Task

Generate complete, runnable Python code for the following TTE design specification.
Use the SOTAstack methodology as your implementation guide and the resolved OMOP concept IDs provided.

## Validated Design Specification

{design_spec}

## Resolved OMOP Concept IDs

{omop_mappings}

## SOTAstack Implementation Reference

{sotastack}

## TTE Rubric (for validation requirements)

{rubric}

## Code Requirements

Generate Python code that implements:

1. **Configuration Constants**
   - Use the RESOLVED OMOP concept IDs from above (not placeholders!)
   - Define concept ID sets for treatments, outcomes, eligibility criteria

2. **Cohort Extraction** (SOTAstack Section 1)
   - SQL queries for OMOP CDM (as strings/templates)
   - Index date identification
   - Eligibility filtering

3. **Feature Engineering** (SOTAstack Section 8)
   - High-dimensional feature extraction
   - Dense features with missing value handling
   - Sparse feature matrix construction

4. **Propensity Score Model** (SOTAstack Section 2)
   - L1 Lasso logistic regression (SAGA solver)
   - Cross-fitted propensity scores

5. **AIPW Estimation** (SOTAstack Section 3)
   - SuperLearner (Lasso + Random Forest)
   - Cross-fitted AIPW with EIF
   - Effect estimation (ATE, RR)

6. **Diagnostics** (SOTAstack Section 4)
   - Balance checks (SMD)
   - Overlap coefficient
   - ESS calculation
   - E-value computation

7. **Negative Control Calibration** (SOTAstack Section 11)
   - Run negative controls
   - Empirical calibration

## Important Notes

- USE the resolved concept IDs from the OMOP Concept ID Mappings section above
- For any terms without matches, use format: `PLACEHOLDER_<description> = None  # TODO: resolve manually`
- Include comments explaining each section
- Make the code modular and well-organized
- Include a `main()` function that orchestrates the full pipeline
- Add docstrings for all functions
- The code should be self-contained and runnable (assuming database connection)

## Output Format

Generate a single Python file with:
1. Imports at the top
2. Configuration constants (with resolved concept IDs)
3. SQL query templates
4. Feature engineering functions
5. Model fitting functions
6. AIPW estimation functions
7. Diagnostic functions
8. Main pipeline orchestration
9. `if __name__ == "__main__":` block

Make the code production-quality with proper error handling.
"""


class CodeAgent:
    """
    Agent that generates Python code for TTE implementation.

    Uses Gemini 3 Pro for code generation.
    """

    def __init__(self, client: GeminiClient):
        """
        Initialize the Code Agent.

        Args:
            client: Gemini client for LLM calls
        """
        self.client = client

    def generate_code(
        self,
        design_spec: str,
        rubric: str,
        sotastack: str,
        omop_mappings: str = "",
    ) -> str:
        """
        Generate Python code for a TTE.

        Args:
            design_spec: The validated design specification
            rubric: Content of rubric.md
            sotastack: Content of SOTAstack.md
            omop_mappings: Formatted OMOP concept ID mappings (optional)

        Returns:
            Python code as a string
        """
        # Default message if no OMOP mappings provided
        if not omop_mappings:
            omop_mappings = (
                "No OMOP lookups performed. Use placeholder concept IDs.\n"
                "Format: PLACEHOLDER_<description> = None  # TODO: resolve manually"
            )

        prompt = CODE_GENERATION_PROMPT.format(
            design_spec=design_spec,
            rubric=rubric,
            sotastack=sotastack,
            omop_mappings=omop_mappings,
        )
        response = self.client.generate(
            prompt,
            temperature=0.3,
            max_output_tokens=30000,
        )

        # Extract code if wrapped in markdown code block
        code = response
        if "```python" in code:
            # Extract code between ```python and ```
            start = code.find("```python") + len("```python")
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()
        elif "```" in code:
            # Try generic code block
            start = code.find("```") + len("```")
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()

        return code
