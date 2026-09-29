"""
Agent 2b: Design Validator

Validates TTE design specifications against the rubric,
checking all decision gates and providing feedback or greenlight.

Uses GPT-5.2 Thinking for structural validation against the rubric.
"""

import json
from typing import Tuple

from ..utils.llm_client import OpenAIClient

VALIDATOR_PROMPT = """You are a validator for Target Trial Emulation (TTE) protocols.

## Design Specification to Validate

{design_spec}

## Validation Rubric (Reference)

{rubric}

## Your Task

Review the design against the rubric. Focus on **structural validation** only:

1. **Time-Zero Alignment** (Gate 4) - The cardinal rule. Verify:
   - Time zero = eligibility time = assignment time = follow-up start
   - No immortal time bias

2. **PICO Completeness** - All components defined (even if imperfect)

3. **Data Mappability** - Can be implemented with OMOP CDM

**Be lenient on:**
- Minor wording issues
- Formatting preferences
- Non-critical methodological details (these can be fixed in code)
- Missing optional components

The goal is to catch fatal design flaws, not achieve perfection.

## Output Format (REQUIRED JSON)

You MUST output valid JSON matching this schema:

{{
  "status": "VALID" or "FEEDBACK",
  "summary": "Brief one-line summary of the design",
  "failed_gates": [],
  "issues": [
    {{
      "gate": "Gate 4",
      "section": "Time-Zero Alignment",
      "severity": "CRITICAL" or "WARNING",
      "issue": "Description of the problem",
      "fix": "Specific action to fix"
    }}
  ],
  "passed_gates": ["Gate 1", "Gate 2", "Gate 3", "Gate 4", "Gate 5", "Gate 6"]
}}

Rules:
- If status is "VALID", failed_gates and issues should be empty arrays
- If status is "FEEDBACK", include failed_gates and issues with specific problems
- Only include CRITICAL issues in failed_gates (not warnings)
- passed_gates should list all gates that passed validation

Output ONLY the JSON. No markdown code blocks, no explanation before or after.
"""


class ValidatorAgent:
    """
    Agent that validates TTE design specifications against the rubric.

    Uses GPT-5.2 Thinking for structural validation (no web search).
    """

    def __init__(self, client: OpenAIClient):
        """
        Initialize the Validator Agent.

        Args:
            client: OpenAI client for LLM calls
        """
        self.client = client

    def validate(self, design_spec: str, rubric: str) -> Tuple[bool, str]:
        """
        Validate a TTE design specification.

        Args:
            design_spec: The design specification from Agent 2a
            rubric: Content of rubric.md

        Returns:
            Tuple of (is_valid, feedback_or_summary)
        """
        prompt = VALIDATOR_PROMPT.format(
            design_spec=design_spec,
            rubric=rubric,
        )
        response = self.client.generate(
            prompt,
            use_web_search=False,  # Structural validation only, no web search
            reasoning_effort="high",
        )

        # Parse JSON response
        return self._parse_validation_response(response)

    def _parse_validation_response(self, response: str) -> Tuple[bool, str]:
        """
        Parse the validator's JSON response.

        Args:
            response: Raw response from LLM

        Returns:
            Tuple of (is_valid, formatted_feedback)
        """
        response_stripped = response.strip()

        # Try to extract JSON from potential markdown code blocks
        if response_stripped.startswith("```"):
            # Remove markdown code block wrapper
            lines = response_stripped.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            response_stripped = "\n".join(lines).strip()

        try:
            result = json.loads(response_stripped)
            is_valid = result.get("status", "").upper() == "VALID"
            return is_valid, json.dumps(result, indent=2)
        except json.JSONDecodeError:
            # Fallback: check for VALID keyword in first 200 chars
            is_valid = "VALID" in response[:200].upper() and "FEEDBACK" not in response[:200].upper()
            return is_valid, response

    def format_feedback_for_revision(self, feedback: str) -> str:
        """
        Format validation feedback for design revision.

        Args:
            feedback: JSON or text feedback from validation

        Returns:
            Formatted feedback with specific issues to fix
        """
        try:
            result = json.loads(feedback)
            if result.get("status", "").upper() == "VALID":
                return "Design is valid. No changes needed."

            issues = result.get("issues", [])
            if not issues:
                return feedback

            lines = ["## Issues to Fix\n"]
            for issue in issues:
                gate = issue.get("gate", "Unknown")
                section = issue.get("section", "")
                problem = issue.get("issue", "")
                fix = issue.get("fix", "")
                severity = issue.get("severity", "CRITICAL")

                lines.append(f"### {gate}: {section}")
                lines.append(f"- **Severity:** {severity}")
                lines.append(f"- **Problem:** {problem}")
                lines.append(f"- **Required Fix:** {fix}")
                lines.append("")

            return "\n".join(lines)
        except json.JSONDecodeError:
            return feedback
