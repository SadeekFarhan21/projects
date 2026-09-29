---
layout: post
title: "Trial Emulation With a Critic That Guards Time Zero"
tab_title: Clinova
code: https://github.com/SadeekFarhan21/projects/tree/main/clinova-trial-emulation
date: 2026-01-25 16:41:36
tags:
  - llm-agents
  - causal-inference
  - healthcare
description: >-
  Clinova is a team-built agent pipeline that drafts a target trial emulation
  from a free-text clinical question, and on the Barrett 2006 benchmark its LLM
  validator rejected two designs over time-zero alignment before accepting the
  third, a loop that took 490 of the run's 837 seconds.
---

Turning a clinical question into an observational study is easy to get subtly wrong, and an easy way to break one is to start a patient's follow-up at the wrong moment. Clinova is a hackathon-style agent pipeline our team built to draft a target trial emulation (TTE)<sup>[[1]](#ref-1)</sup> from a question in plain English: one agent turns the question into a causal PICO, a second drafts the trial protocol, a third audits the protocol against a written rubric, a lookup tool maps clinical terms to OMOP concept IDs<sup>[[6]](#ref-6)</sup>, and a last agent writes Python for the All of Us enclave.

The interesting part is the critic. On a benchmark against the Barrett 2006 trial of two contrast agents in chronic kidney disease<sup>[[7]](#ref-7)</sup>, **the validator rejected the first design, rejected the second, and accepted the third, and both rejections were about when follow-up starts.** The second rejection caught a bias the design agent introduced while fixing the first. The whole run took **837 seconds**, and **490** of them went to the validate-and-revise loop.

## Why It Matters

Target trial emulation asks for discipline that is easy to state and easy to break. Before touching data you write down the eligibility criteria, the treatment strategies, the assignment procedure, the outcome, the follow-up, the causal contrast and the analysis plan, and you make time zero, the start of follow-up, coincide with the moment eligibility is met and treatment is assigned<sup>[[2]](#ref-2)</sup>. Getting that wrong produces bias that no estimator fixes afterwards.

The idea of Clinova was to let language models do the drafting and make a second model do the checking. A researcher puts a clinical question in plain English on one end and gets runnable analysis code on the other, with a written rubric in between so the critic checks the design against something other than its own taste.

We picked five published contrast-nephropathy trials as benchmark targets, so that the pipeline's designs could be compared with designs real investigators had already committed to. Barrett is the one this post follows end to end. Praneel Patel added the Barrett run folder this post reads from, along with the example folders for the website. My part was the `GET /api/examples` endpoint in `server.py` and the frontend integration and UI.

## Technical Details

### Time Zero Is the Cardinal Rule

In a randomized trial, eligibility, randomization and the start of follow-up happen at the same instant. In observational data they don't, and the ways they drift apart have names. If a patient must survive until they start treatment to be counted as treated, the treated group is handed guaranteed event-free time, which is immortal time bias<sup>[[3]](#ref-3)</sup>. If follow-up starts after treatment has already begun, early events are missed, which is left truncation. If eligibility is decided using information from after time zero, the cohort is selected on the future.

Our `rubric.md`, 638 lines long, turns this into a check the validator can apply. Three timestamps must coincide, eligibility, assignment and the start of follow-up, and every eligibility rule must be evaluable from information available at or before that timestamp.

### The Pipeline

The pipeline is a straight chain with one loop.

<figure class="excal" data-diagram="clinova-trial-emulation-pipeline"><a href="/img/diagrams/clinova-trial-emulation-pipeline.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/clinova-trial-emulation-pipeline.webp" alt="Clinova pipeline: a free-text clinical question flows through the Question, Design and Validator agents on gpt-5.2, with a dashed feedback arrow from the validator back to the design agent for at most 3 revise_spec iterations, then a local OMOP concept lookup and the Code agent on gemini-3-pro-preview, which produces analysis code for the All of Us enclave." width="2400" height="1373" loading="lazy" decoding="async"></a></figure>

The question, design and validator agents are configured for `gpt-5.2` in `agent/config.py`, and the same file sets `GEMINI_MODEL = "gemini-3-pro-preview"` for the code agent. The OMOP lookup is a local tool with no LLM call. The validator can send the design back to the design agent at most three times. For Barrett, the code agent's output was a 470-line analysis script.

### Why a Critic Instead of a Better Prompt

A design spec is a long document. The Barrett one is 341 lines in its first version, and a subtle error in a window definition on line 200 is easy to write and easy to miss. Asking the same model to be more careful is a weak lever. A separate call whose only job is to apply a fixed list of gates, with structured output, gives a checkable signal, and its JSON feeds straight into a targeted revision prompt. The loop is bounded, because a critic can also oscillate.

Here is what that looked like on Barrett.

<figure class="excal" data-diagram="clinova-trial-emulation-validator-loop"><a href="/img/diagrams/clinova-trial-emulation-validator-loop.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/clinova-trial-emulation-validator-loop.webp" alt="Three design spec drafts and the validator's verdicts on the Barrett 2006 run: v1 (t0 = contrast administration time) gets FEEDBACK with 3 CRITICAL and 1 WARNING issues, v2 (t0 = max of procedure and drug start) gets 1 CRITICAL left-truncation issue, and v3 (t0 = drug_exposure_start_datetime) is VALID with 6 of 6 gates passed." width="2400" height="1534" loading="lazy" decoding="async"></a></figure>

### From Free Text to Concept IDs

Analysis code needs concept IDs from the OMOP vocabulary, not phrases like "iodixanol" or "multidetector computed tomography". The lookup tool is a tiered search. It tries an exact match on normalized text, then a RapidFuzz fuzzy match with an early exit when the score is at least 95, then a semantic search over SapBERT<sup>[[4]](#ref-4)</sup> embeddings stored in a FAISS index<sup>[[5]](#ref-5)</sup>. Ties are broken with per-domain vocabulary weights, for example RxNorm for drugs and LOINC for measurements. On Barrett, the exact tier resolved `iodixanol` to 19003201 and `iopamidol` to 19081224, both RxNorm at confidence 1.0.

## Implementation

### The Validation Loop

The loop in `agent/main.py` is short. Validate, and if the answer isn't VALID, format the issues and ask the design agent for a targeted revision rather than a fresh draft.

```python
while not is_valid and iteration < MAX_VALIDATION_ITERATIONS:
    iteration += 1
    is_valid, feedback = validator_agent.validate(design_spec, rubric)
    all_feedback.append(f"=== Iteration {iteration} ===\n{feedback}")
    if not is_valid:
        formatted_feedback = validator_agent.format_feedback_for_revision(feedback)
        design_spec = design_agent.revise_spec(
            causal_question=causal_question,
            previous_spec=design_spec,
            feedback=formatted_feedback,
            rubric=rubric,
            sotastack=sotastack,
        )
```

### A Validator That Returns JSON

The validator prompt asks for three things, time-zero alignment, PICO completeness and OMOP mappability, and tells the model to be lenient about wording and non-critical methodology so it hunts for fatal flaws and not for polish. The output is fixed JSON.

```json
{
  "status": "VALID" or "FEEDBACK",
  "failed_gates": [],
  "issues": [
    { "gate": "Gate 4", "section": "Time-Zero Alignment",
      "severity": "CRITICAL" or "WARNING",
      "issue": "Description of the problem",
      "fix": "Specific action to fix" }
  ],
  "passed_gates": ["Gate 1", "Gate 2", "Gate 3", "Gate 4", "Gate 5", "Gate 6"]
}
```

The `fix` field matters as much as the `issue` field. `format_feedback_for_revision` turns each issue into a block with the problem and a required fix, and that block is all the revising agent sees of the critique.

### The Tiered Lookup

The search function returns as soon as a cheap tier is confident, and only falls through to the expensive one when it has to.

```python
exact_results = self._exact_match(normalized_query, domain_filter)
if exact_results:
    ...  # confidence 1.0, return immediately

fuzzy_results = self._fuzzy_match(normalized_query, tokens, domain_filter)
if fuzzy_results:
    best_fuzzy = fuzzy_results[0]
    if best_fuzzy["score"] >= self.config.fuzzy_high_confidence:  # 95.0
        ...  # return immediately

# Tier 3: semantic search, then merge and rank
```

The thresholds live in `omop_lookup/config.py`: 0.85 for a strong semantic match and 0.60 as the floor, 95 and 70 for fuzzy, and 0.75 as the confidence below which the tool returns candidates instead of a single answer.

## Problems

### 1. The First Draft Selected Patients on the Future

The first Barrett design read fine and was wrong in three places. **The validator's first pass found three CRITICAL time-zero issues, all of them windows that looked past the start of follow-up.** The eligibility rule linked a qualifying CT to the contrast drug record with a window of up to 6 hours after time zero, so whether a patient was eligible depended on the future. The mixed-exposure exclusion looked forward from 1 hour before to 6 hours after time zero. The intra-arterial exclusion used a window of zero to one day around time zero. It also raised one Gate 3 WARNING about datasets that record dates and not datetimes.

Each is a subtle window definition, the kind of error that's easy to write and easy to miss in a 341-line spec.

### 2. Fixing One Bias Introduced Another

For the eligibility issue, the critic suggested `t0 = min(CT procedure_datetime, contrast drug_exposure_start_datetime)`, or requiring the procedure time to be at or before the drug start plus a small tolerance. The design agent picked `max()` instead: `t0 = max(procedure_datetime, drug_exposure_start_datetime)`, so that both pieces of evidence sit at or before time zero. That satisfies the eligibility complaint and creates a new bug. If the drug was recorded before the CT, treatment starts before time zero, which is left truncation.

**The second validation pass flagged exactly that as its only CRITICAL issue** and recommended `t0 = drug_exposure_start_datetime`. Version 3 adopted it, and the third pass returned VALID with all six gates passed.

So the loop converged, and it needed all three rounds, because the revision that fixed one violation introduced another. **A single pass of critique would have shipped version 2.** That's the case for a bounded loop over a one-shot review.

## Results

The numbers below come from `analysis/summarize_barrett_run.py`, a small offline script I wrote that parses the Barrett artifacts: stage durations from the timestamps in `run_log.txt`, validator issues per iteration from the JSON blocks in `step2b_validation_feedback.txt`, and match types and confidences from `step2c_omop_results.csv`. It writes `results/barrett_run_summary.json`.

### Where the Time Goes

<figure data-figure="chart:projects/clinova-trial-emulation/clinova-trial-emulation-stage-time"></figure>

The run took 837 seconds, from 04:29:47 to 04:43:44 on 2026-01-25. **The validation loop was 490 seconds of that, about 59 percent**, over three iterations of 216, 221 and 53 seconds. Iterations 1 and 2 each include a design revision, and iteration 3 is a validate call alone, so a lone validate takes about 53 seconds. The design agent took 173 seconds, the question agent 87, the code agent 81, and the OMOP lookup 6, because it's a local lookup with no model call. **The validate-and-revise loop is the most expensive stage in the pipeline.**

### The Validator Converged in Three Rounds

<figure data-figure="chart:projects/clinova-trial-emulation/clinova-trial-emulation-validator-issues"></figure>

Iteration 1 returned four issues, three CRITICAL and one WARNING. Iteration 2 returned one CRITICAL. **Iteration 3 returned none and passed 6 of 6 gates.** The design spec grew from 341 lines to 381 to 385.

### Drug Names Resolve Exactly

<figure data-figure="chart:projects/clinova-trial-emulation/clinova-trial-emulation-top1-matches"></figure>

Of the 20 query terms in the results file, 6 were exact name matches, 5 exact synonym matches and 9 fuzzy name matches. **Fourteen top-ranked hits have confidence of at least 0.95**, and the two contrast agents at the center of the trial, iodixanol and iopamidol, both resolve exactly to their RxNorm concepts.

## What I Would Change

### Make the Critic Verify, Not Only Suggest

The critic suggested `min()` and the reviser chose `max()`, which introduced a new violation, and "converged" here means an LLM said VALID. A cheap improvement is a deterministic pre-check that parses each time-zero definition and tests the three-way coincidence rule on synthetic timestamps, with the LLM critic left for what can't be parsed. Both failure modes in the Barrett run, look-forward windows and treatment before time zero, are checkable that way.

### Run All Five Benchmarks, More Than Once

Barrett shows what a full run should leave behind. Running all five trials, repeating each to see how often the validator converges and whether the same critique shows up twice, and comparing the designs and results with the published effect sizes would turn one good run into an evaluation.

## References

1. <span id="ref-1"></span>Hernán MA, Robins JM. *Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available*. American Journal of Epidemiology 183(8):758-764, 2016. [link](https://doi.org/10.1093/aje/kwv254)
2. <span id="ref-2"></span>Hernán MA, Sauer BC, Hernández-Díaz S, Platt R, Shrier I. *Specifying a target trial prevents immortal time bias and other self-inflicted injuries in observational analyses*. Journal of Clinical Epidemiology 79:70-75, 2016. [link](https://doi.org/10.1016/j.jclinepi.2016.04.014)
3. <span id="ref-3"></span>Suissa S. *Immortal Time Bias in Pharmacoepidemiology*. American Journal of Epidemiology 167(4):492-499, 2008. [link](https://doi.org/10.1093/aje/kwm324)
4. <span id="ref-4"></span>Liu F, Shareghi E, Meng Z, Basaldella M, Collier N. *Self-Alignment Pretraining for Biomedical Entity Representations*. NAACL 2021. [link](https://arxiv.org/abs/2010.11784)
5. <span id="ref-5"></span>Johnson J, Douze M, Jégou H. *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 2019. [link](https://arxiv.org/abs/1702.08734)
6. <span id="ref-6"></span>Hripcsak G, Duke JD, Shah NH, Reich CG, Huser V, et al. *Observational Health Data Sciences and Informatics (OHDSI): Opportunities for Observational Researchers*. Studies in Health Technology and Informatics 216:574-578, 2015. [link](https://pubmed.ncbi.nlm.nih.gov/26262116/)
7. <span id="ref-7"></span>Barrett BJ, Katzberg RW, Thomsen HS, Chen N, Sahani D, Soulez G, et al. *Contrast-induced nephropathy in patients with chronic kidney disease undergoing computed tomography: a double-blind comparison of iodixanol and iopamidol*. Investigative Radiology 41(11):815-821, 2006. [link](https://pubmed.ncbi.nlm.nih.gov/17035872/)
