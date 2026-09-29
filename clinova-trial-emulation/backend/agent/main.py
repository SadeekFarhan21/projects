#!/usr/bin/env python3
"""
Virtual Clinical Trial Agent - Main Entry Point

Orchestrates the multi-agent pipeline to transform a medical question
into a Target Trial Emulation design and implementation code.

Pipeline:
1. Question Agent (GPT-5.2 Thinking + Search): Transform question to causal PICO
2. Design Agent (GPT-5.2 Thinking + Search): Generate TTE protocol spec
3. Validator Agent (GPT-5.2 Thinking + Search): Validate spec (loop up to 3x)
4. OMOP Lookup Agent: Resolve medical terms to concept IDs
5. Code Agent (Gemini 3 Pro): Generate Python implementation

Usage:
    python -m agent
    python -m agent --question "your custom question here"
"""

import argparse
import sys

from .config import (
    get_openai_api_key,
    get_gemini_api_key,
    MAX_VALIDATION_ITERATIONS,
)
from .agents import (
    QuestionAgent,
    DesignAgent,
    ValidatorAgent,
    CodeAgent,
    OMOPLookupAgent,
)
from .utils import (
    OpenAIClient,
    GeminiClient,
    create_run_folder,
    save_step_output,
    append_to_log,
    load_rubric,
    load_sotastack,
)


# Default sample input
SAMPLE_QUESTION = (
    "how are clinically relevant eGFR subgroups affected differently "
    "by iodinated contrast agents & risk of AKI in the next 30 days"
)


def run_pipeline(question: str) -> str:
    """
    Run the full TTE agent pipeline.

    Args:
        question: The natural text medical question

    Returns:
        Path to the run folder containing all outputs
    """
    print("=" * 60)
    print("Virtual Clinical Trial Agent")
    print("=" * 60)
    print(f"\nInput Question: {question}\n")

    # Create run folder
    run_folder = create_run_folder(question)
    print(f"Run folder: {run_folder}\n")

    # Save input
    save_step_output(run_folder, "input.txt", question)
    append_to_log(run_folder, f"Started pipeline with question: {question}")

    # Load context files
    print("Loading context files...")
    rubric = load_rubric()
    sotastack = load_sotastack()
    append_to_log(run_folder, "Loaded rubric.md and SOTAstack.md")

    # Initialize clients
    print("Initializing LLM clients...")
    openai_client = OpenAIClient(get_openai_api_key())
    gemini_client = GeminiClient(get_gemini_api_key())
    append_to_log(run_folder, "Initialized OpenAI and Gemini clients")

    # Initialize agents
    question_agent = QuestionAgent(openai_client)
    design_agent = DesignAgent(openai_client)
    validator_agent = ValidatorAgent(openai_client)
    omop_agent = OMOPLookupAgent(top_k=10)
    code_agent = CodeAgent(gemini_client)

    # =========================================================================
    # Step 1: Transform question to causal question
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 1: Transforming question to causal question...")
    print("        (GPT-5.2 Thinking with Search)")
    print("-" * 60)
    append_to_log(run_folder, "Starting Step 1: Question transformation")

    causal_question = question_agent.transform(question)
    save_step_output(run_folder, "step1_causal_question.txt", causal_question)
    append_to_log(run_folder, "Completed Step 1: Saved causal question")
    print("\nCausal question generated and saved.")

    # =========================================================================
    # Step 2a: Generate design spec
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 2a: Generating TTE design specification...")
    print("         (GPT-5.2 Thinking with Search)")
    print("-" * 60)
    append_to_log(run_folder, "Starting Step 2a: Design specification")

    design_spec = design_agent.generate_spec(causal_question, rubric, sotastack)
    save_step_output(run_folder, "step2_design_spec.md", design_spec)
    append_to_log(run_folder, "Completed Step 2a: Saved design specification")
    print("\nDesign specification generated and saved.")

    # =========================================================================
    # Step 2b: Validate design (with iteration loop)
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 2b: Validating design specification...")
    print("         (GPT-5.2 Thinking with Search)")
    print("-" * 60)
    append_to_log(run_folder, "Starting Step 2b: Design validation")

    iteration = 0
    is_valid = False
    all_feedback = []

    while not is_valid and iteration < MAX_VALIDATION_ITERATIONS:
        iteration += 1
        print(f"\nValidation iteration {iteration}/{MAX_VALIDATION_ITERATIONS}")
        append_to_log(run_folder, f"Validation iteration {iteration}")

        is_valid, feedback = validator_agent.validate(design_spec, rubric)
        all_feedback.append(f"=== Iteration {iteration} ===\n{feedback}")

        if is_valid:
            print("Design VALIDATED!")
            append_to_log(run_folder, "Design validated successfully")
        else:
            print("Design needs revision. Revising with targeted feedback...")
            append_to_log(run_folder, "Validation failed, revising design with feedback")

            # Format feedback for revision (handles JSON or plain text)
            formatted_feedback = validator_agent.format_feedback_for_revision(feedback)

            # Revise design with previous spec + structured feedback
            design_spec = design_agent.revise_spec(
                causal_question=causal_question,
                previous_spec=design_spec,
                feedback=formatted_feedback,
                rubric=rubric,
                sotastack=sotastack,
            )
            save_step_output(
                run_folder,
                f"step2_design_spec_v{iteration + 1}.md",
                design_spec,
            )

    # Save all validation feedback
    save_step_output(
        run_folder,
        "step2_validator_feedback.txt",
        "\n\n".join(all_feedback),
    )

    if not is_valid:
        print(f"\nWarning: Design not fully validated after {MAX_VALIDATION_ITERATIONS} iterations.")
        print("Proceeding with best available design...")
        append_to_log(run_folder, "Warning: Proceeding with unvalidated design")

    # Save final design spec
    save_step_output(run_folder, "step2_design_spec.md", design_spec)

    # =========================================================================
    # Step 2c: OMOP Concept Lookup
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 2c: Resolving medical terms to OMOP concept IDs...")
    print("         (Local OMOP Lookup Tool)")
    print("-" * 60)
    append_to_log(run_folder, "Starting Step 2c: OMOP concept lookup")

    omop_mappings = ""

    # Extract terms from design spec
    lookup_terms = design_agent.extract_lookup_terms(design_spec)

    if lookup_terms:
        print(f"\nFound {len(lookup_terms)} terms to look up:")
        for term, domain in lookup_terms[:10]:  # Show first 10
            print(f"  - {term} ({domain or 'any domain'})")
        if len(lookup_terms) > 10:
            print(f"  ... and {len(lookup_terms) - 10} more")

        # Save lookup terms file
        omop_agent.save_lookup_terms_file(
            lookup_terms,
            f"{run_folder}/step2c_lookup_terms.txt",
        )

        # Perform lookups (gracefully handles missing database)
        print("\nPerforming OMOP lookups...")
        lookup_results = omop_agent.lookup_terms(lookup_terms)

        # Save results
        omop_agent.save_results_csv(
            lookup_results,
            f"{run_folder}/step2c_omop_results.csv",
        )

        # Format for code agent
        omop_mappings = omop_agent.format_for_code_agent(lookup_results)
        save_step_output(run_folder, "step2c_omop_mappings.md", omop_mappings)

        # Count successful lookups
        successful = sum(1 for matches in lookup_results.values() if matches)
        print(f"\nOMOP lookup complete: {successful}/{len(lookup_terms)} terms resolved")
        append_to_log(
            run_folder,
            f"Completed Step 2c: {successful}/{len(lookup_terms)} terms resolved",
        )
    else:
        print("\nNo OMOP lookup terms found in design spec.")
        append_to_log(run_folder, "Step 2c: No lookup terms found")

    # =========================================================================
    # Step 3: Generate code
    # =========================================================================
    print("\n" + "-" * 60)
    print("STEP 3: Generating Python code...")
    print("        (Gemini 3 Pro)")
    print("-" * 60)
    append_to_log(run_folder, "Starting Step 3: Code generation")

    code = code_agent.generate_code(design_spec, rubric, sotastack, omop_mappings)
    save_step_output(run_folder, "step3_code.py", code)
    append_to_log(run_folder, "Completed Step 3: Saved Python code")
    print("\nPython code generated and saved.")

    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {run_folder}")
    print("\nGenerated files:")
    print("  - input.txt                    (original question)")
    print("  - step1_causal_question.txt    (PICO causal question)")
    print("  - step2_design_spec.md         (TTE protocol specification)")
    print("  - step2_validator_feedback.txt (validation results)")
    print("  - step2c_lookup_terms.txt      (terms for OMOP lookup)")
    print("  - step2c_omop_results.csv      (OMOP lookup results)")
    print("  - step2c_omop_mappings.md      (formatted concept IDs)")
    print("  - step3_code.py                (Python TTE implementation)")
    print("  - run_log.txt                  (execution log)")

    append_to_log(run_folder, "Pipeline completed successfully")
    return run_folder


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Virtual Clinical Trial Agent - TTE Design Pipeline"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=SAMPLE_QUESTION,
        help="The medical question to process (default: sample eGFR/contrast/AKI question)",
    )

    args = parser.parse_args()

    try:
        run_folder = run_pipeline(args.question)
        print(f"\nSuccess! Check {run_folder} for results.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
