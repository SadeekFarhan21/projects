#!/usr/bin/env python3
"""
GPT-5.2-Thinking Paper Reader for TTE Agent Development

Reads academic papers using GPT-5.2 with web search and high reasoning effort,
extracting notes relevant to building our virtual clinical trial agent.
"""

import os
import sys
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

RUBRIC_PROMPT = """
## Context: What We're Building

We are building a virtual clinical trial agent system with two agents:
1. **trialRunnerAgent**: Plans, codes, runs, and assembles reports on virtual clinical trials
   (Target Trial Emulations / TTEs) given a natural-text medical question
2. **medicalAgent**: Wraps the first agent to identify medical questions that could benefit
   from TTE analysis

The trialRunnerAgent needs:
- A rubric for what makes a valid TTE design
- Understanding of biases to avoid (immortal time, confounding, selection)
- Knowledge of SOTA methodology (IPTW-PS, hazard ratios, agreement metrics)
- Guidelines for when TTEs can/cannot reliably answer a question

Our goal is to translate the TTE framework from academic labs into automated software
that can design and run TTEs with minimal human intervention, achieving regulatory-grade
accuracy comparable to RCT-DUPLICATE benchmarks.

---

## Your Task

Search for and read this paper: {paper_title} (DOI: {doi})

Extract notes relevant to our TTE agent development:

1. **Goal**: What is this paper trying to achieve?
2. **TTE Relevance**: Does it inform what makes a valid trial design? What requirements/standards does it specify?
3. **Bias Considerations**: What does it say about immortal time bias, confounding, selection bias, or other pitfalls?
4. **Methodology**: What statistical methods, estimands, or agreement metrics does it recommend?
5. **Design Requirements**: What does a good TTE need according to this paper? Any checklists or protocols?
6. **Agent Implications**: How should this inform our automated trialRunnerAgent? What rules/checks should the agent implement?

Be specific, cite key points from the paper, and focus on actionable insights for building our agent.
"""


def read_paper(paper_title: str, doi: str) -> str:
    """
    Use GPT-5.2 with web search and high reasoning to read and extract notes from a paper.

    Args:
        paper_title: Full title/citation of the paper
        doi: DOI of the paper

    Returns:
        Structured notes from the paper
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-5.2",
        tools=[{"type": "web_search"}],
        reasoning={"effort": "high"},
        input=RUBRIC_PROMPT.format(paper_title=paper_title, doi=doi)
    )

    return response.output_text


def main():
    parser = argparse.ArgumentParser(
        description="Read academic papers using GPT-5.2 with web search for TTE agent development"
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Paper title/citation"
    )
    parser.add_argument(
        "--doi",
        required=True,
        help="Paper DOI"
    )
    parser.add_argument(
        "--output",
        help="Optional output file (defaults to stdout)"
    )

    args = parser.parse_args()

    try:
        notes = read_paper(args.title, args.doi)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(notes)
            print(f"Notes written to {args.output}")
        else:
            print(notes)

    except Exception as e:
        print(f"Error processing paper: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
