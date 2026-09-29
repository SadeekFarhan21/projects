"""
Agent 2a: Design Spec Generator

Generates a complete TTE protocol specification from a causal question,
using the rubric and SOTA methodology as guidance.

Also outputs a list of medical terms that need OMOP concept ID lookup.
"""

import re
from typing import Tuple

from ..utils.llm_client import OpenAIClient

DESIGN_SPEC_PROMPT = """You are an expert in Target Trial Emulation (TTE) study design.

## Your Task

Design a complete TTE protocol specification for the following causal question.
Use the rubric below as your template and the SOTA methodology for implementation guidance.

## Causal Question

{causal_question}

## TTE Design Rubric

{rubric}

## SOTA Methodology Reference

{sotastack}

## Instructions

Generate a COMPLETE TTE protocol specification that includes:

1. **Eligibility Criteria** (Part I, Component 1)
   - Specific inclusion/exclusion criteria
   - How each criterion maps to OMOP data
   - Lookback windows for each

2. **Treatment Strategies** (Part I, Component 2)
   - Intervention definition with concept IDs (use placeholders if needed)
   - Comparator definition with concept IDs
   - Assignment rules

3. **Treatment Assignment** (Part I, Component 3)
   - New-user design specification
   - Confounder list for adjustment
   - Washout period

4. **Outcomes** (Part I, Component 4)
   - Primary outcome definition with concept IDs
   - Risk window (e.g., 30 days)
   - Validation status

5. **Time Zero Definition** (Part I, Component 5)
   - Index date definition
   - Three-way coincidence verification

6. **Follow-up** (Part I, Component 6)
   - Censoring events
   - Maximum follow-up period

7. **Estimand** (Part I, Component 7)
   - ITT analogue vs per-protocol
   - Effect measure (RD, RR, HR)
   - ICE table

8. **Analysis Plan** (Part I, Component 8)
   - Primary method (recommend AIPW per SOTAstack)
   - Sensitivity analyses
   - Diagnostic thresholds

## IMPORTANT: OMOP Lookup Terms

At the END of your response, include a section titled "## OMOP_LOOKUP_TERMS" with a list of all medical terms that need to be looked up in OMOP/Athena to get concept IDs.

Format each term on its own line with the domain hint:
```
term|domain
```

Where domain is one of: Condition, Procedure, Drug, Measurement, Observation, Device

Example:
```
## OMOP_LOOKUP_TERMS
acute kidney injury|Condition
iodinated contrast media|Drug
CT scan with contrast|Procedure
serum creatinine|Measurement
eGFR|Measurement
chronic kidney disease|Condition
```

Include ALL terms that need OMOP concept IDs for:
- Eligibility criteria (conditions, procedures, measurements)
- Treatment/exposure definitions
- Outcome definitions
- Covariate/confounder definitions

## Output Format

Generate a Markdown document with all protocol components clearly specified.
Use tables where appropriate. Be specific about data mappings to OMOP CDM.

For now, use placeholder format like `[OMOP: term_name]` where concept IDs are needed.
These will be resolved by the OMOP lookup tool before code generation.
"""

DESIGN_REVISION_PROMPT = """You are revising a TTE design specification to fix validation issues.

## Previous Design Specification

{previous_spec}

## Validation Feedback (Issues to Fix)

{feedback}

## Your Task

Fix ONLY the specific issues listed above. Do NOT regenerate from scratch.
Keep all working sections unchanged. Make targeted edits.

## Causal Question (for reference)

{causal_question}

## TTE Design Rubric

{rubric}

## SOTA Methodology Reference

{sotastack}

## Instructions

1. Review the feedback carefully - each issue has a gate, problem description, and required fix
2. Make the minimum changes needed to address each issue
3. Preserve all sections that are NOT mentioned in the feedback
4. Ensure the ## OMOP_LOOKUP_TERMS section is preserved and updated if needed

## Output

Return the COMPLETE revised design specification (not just the changes).
The output should be a full Markdown document ready for re-validation.
"""


class DesignAgent:
    """
    Agent that generates TTE protocol specifications from causal questions.

    Also extracts medical terms for OMOP concept ID lookup.
    """

    def __init__(self, client: OpenAIClient):
        """
        Initialize the Design Agent.

        Args:
            client: OpenAI client for LLM calls
        """
        self.client = client

    def generate_spec(
        self,
        causal_question: str,
        rubric: str,
        sotastack: str,
    ) -> str:
        """
        Generate a TTE design specification.

        Args:
            causal_question: The causal question from Agent 1
            rubric: Content of rubric.md
            sotastack: Content of SOTAstack.md

        Returns:
            The TTE protocol specification in Markdown format
        """
        prompt = DESIGN_SPEC_PROMPT.format(
            causal_question=causal_question,
            rubric=rubric,
            sotastack=sotastack,
        )
        response = self.client.generate(
            prompt,
            use_web_search=True,
            reasoning_effort="high",
        )
        return response

    def revise_spec(
        self,
        causal_question: str,
        previous_spec: str,
        feedback: str,
        rubric: str,
        sotastack: str,
    ) -> str:
        """
        Revise a TTE design specification based on validation feedback.

        This is a targeted revision mode that fixes specific issues rather
        than regenerating from scratch.

        Args:
            causal_question: The original causal question
            previous_spec: The previous design specification that failed validation
            feedback: Formatted feedback with specific issues to fix
            rubric: Content of rubric.md
            sotastack: Content of SOTAstack.md

        Returns:
            The revised TTE protocol specification in Markdown format
        """
        prompt = DESIGN_REVISION_PROMPT.format(
            previous_spec=previous_spec,
            feedback=feedback,
            causal_question=causal_question,
            rubric=rubric,
            sotastack=sotastack,
        )
        response = self.client.generate(
            prompt,
            use_web_search=False,  # Revision doesn't need web search
            reasoning_effort="high",
        )
        return response

    def extract_lookup_terms(self, design_spec: str) -> list[tuple[str, str]]:
        """
        Extract OMOP lookup terms from the design specification.

        Uses flexible matching to handle various header formats and term syntaxes.

        Args:
            design_spec: The design specification markdown

        Returns:
            List of (term, domain) tuples for OMOP lookup
        """
        terms = []

        # Case-insensitive, flexible header matching
        header_patterns = [
            r"##\s*OMOP_LOOKUP_TERMS",
            r"##\s*OMOP\s+Lookup\s+Terms",
            r"##\s*Concept\s+Sets",
            r"##\s*Terms\s+for\s+OMOP",
            r"##\s*OMOP\s+Terms",
            r"##\s*Medical\s+Terms\s+for\s+Lookup",
        ]

        section = None
        for pattern in header_patterns:
            match = re.search(pattern, design_spec, re.IGNORECASE)
            if match:
                section = design_spec[match.end():]
                break

        if not section:
            print("WARNING: No OMOP lookup section found in design spec")
            return terms

        # Stop at next section header
        if "\n## " in section:
            section = section.split("\n## ")[0]

        # Parse each line with multiple format support
        for line in section.strip().split("\n"):
            line = line.strip()

            # Skip empty lines, code blocks, headers
            if not line or line.startswith("```") or line.startswith("#"):
                continue

            # Remove markdown list markers
            if line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("* "):
                line = line[2:].strip()

            term, domain = None, None

            # Format 1: term|domain (primary format)
            if "|" in line:
                parts = line.split("|")
                term = parts[0].strip()
                domain = parts[1].strip() if len(parts) > 1 else None

            # Format 2: term - domain
            elif " - " in line:
                parts = line.split(" - ", 1)
                term = parts[0].strip()
                domain = parts[1].strip() if len(parts) > 1 else None

            # Format 3: term (domain)
            elif "(" in line and ")" in line:
                match = re.match(r"(.+?)\s*\((\w+)\)\s*$", line)
                if match:
                    term = match.group(1).strip()
                    domain = match.group(2).strip()
                else:
                    term = line

            # Format 4: term: domain
            elif ": " in line:
                parts = line.split(": ", 1)
                term = parts[0].strip()
                domain = parts[1].strip() if len(parts) > 1 else None

            else:
                # Plain term without domain
                term = line

            # Clean up term (remove any remaining markers)
            if term:
                term = term.strip("- *`")
                if term:
                    terms.append((term, domain))

        if not terms:
            print("WARNING: OMOP lookup section found but no terms parsed")

        return terms
