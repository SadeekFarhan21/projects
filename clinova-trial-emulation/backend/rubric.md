# Target Trial Emulation Rubric

This rubric defines what makes a valid Target Trial Emulation (TTE). It is the core reference for the spec agent, analysis agent, and code agent to ensure regulatory-grade accuracy comparable to RCT-DUPLICATE benchmarks.

---

## Spec → Code Quick Reference

> **For effectiveness and safety trials, use the simplified pathway below.**
> Complex designs (per-protocol, dynamic regimes, clone-censor-weight) are rarely needed.

### Recommended Design: Active Comparator New-User (ACNU) + AIPW

| Spec Component | Code Implementation | SOTAstack Reference |
|----------------|---------------------|---------------------|
| **Eligibility** | SQL WHERE clauses on OMOP tables | §1: Query Extraction |
| **Treatment strategies** | `procedure_concept_id IN (...)` | §12: Clinical Procedure Groups |
| **Time zero (index date)** | `procedure_datetime` | §1A: Cohort Identification |
| **Outcomes** | `condition_concept_id` + time window | §5: Endpoint Enumeration |
| **Confounders** | High-dim features from all OMOP domains | §1D + §8: Feature Engineering |
| **Propensity model** | L1 Lasso (SAGA solver) | §2: LSPS Model |
| **Effect estimation** | Cross-fitted AIPW with SuperLearner | §3: SuperLearner AIPW |
| **Balance diagnostics** | Sparse SMD calculation | §4B: SMD Calculation |
| **Sensitivity analysis** | Negative control calibration | §4D + §11: Negative Controls |

### Simplified Decision Flow (Effectiveness/Safety)

```
1. DEFINE: Treatment vs Comparator (procedure/drug codes)
   → SOTAstack §12: Use existing procedure groups or define new

2. DEFINE: Index date = first qualifying procedure/prescription
   → SOTAstack §1A: procedure_datetime or drug_exposure_start_date

3. DEFINE: Outcome + window (e.g., AKI within 30 days)
   → SOTAstack §5: condition_concept_id + window_days

4. EXTRACT: High-dimensional baseline features
   → SOTAstack §1D + §8: All OMOP domains before index

5. FIT: Propensity score (L1 Lasso)
   → SOTAstack §2: Cross-fitted, C=0.2, SAGA solver

6. TRIM: PS extremes [0.025, 0.975]
   → SOTAstack §9: Trimming

7. RUN: AIPW with SuperLearner
   → SOTAstack §3: 5-fold cross-fitting

8. CHECK: Diagnostics
   → SOTAstack §4: SMD < 0.1, ESS/N > 0.5, overlap > 0.5

9. CALIBRATE: Using negative controls
   → SOTAstack §4D + §11: Empirical null calibration
```

### When to Use Complex Designs

| Scenario | Design | When NOT Needed |
|----------|--------|-----------------|
| Grace period for initiation | Clone-censor-weight | If index = first prescription/procedure (instant) |
| Compare "initiate vs never" | Sequential trials | If using active comparator |
| Effect of adherence | Per-protocol + g-methods | If estimating initiation effect (ITT analogue) |
| Dynamic treatment rules | Clone-censor-weight | If comparing fixed strategies |

**For most effectiveness/safety questions:** ACNU + AIPW is sufficient.

---

## Part I: Major Protocol Components (Decision Gates)

Each component below MUST be explicitly defined and validated before proceeding with the TTE.

> "Table 1 lists **7 key components** of the target trial protocol…" — Hernán & Robins 2016

| Component | Description | Decision Gate (Validity Criteria) | Required Fields |
|-----------|-------------|-----------------------------------|-----------------|
| **1. Eligibility Criteria** | Defines WHO enters the trial. Must specify the population characteristics, inclusion/exclusion criteria, and the clinical context. | **VALID IF:** (1) All criteria are measurable in the data source, (2) Criteria are evaluated ONLY at baseline (time zero), (3) Criteria reflect a pragmatic clinical pathway. **INVALID IF:** Criteria reference future events (e.g., "must survive 90 days," "must have 2 follow-up labs"). | `{criterion, data_mapping, lookback_window, code_list, code_list_version}` for each |
| **2. Treatment Strategies** | Defines WHAT interventions are being compared. Must specify treatment(s), comparator(s), dosing, timing, duration, and adherence/deviation rules. | **VALID IF:** (1) Strategies are well-defined interventions, (2) Active comparator preferred, (3) Strategies distinguishable at time zero, (4) Grace periods handled via clone-censor-weight. **INVALID IF:** Strategies use future treatment behavior (e.g., "received ≥3 refills"). | `{initiation_rule, grace_period, adherence_definition, deviation_rules, allowed_switching}` |
| **3. Treatment Assignment** | Defines HOW individuals are assigned to strategies. In observational data, based on observed treatment adjusted for confounding. | **VALID IF:** (1) Assignment at time zero, (2) Uses ONLY baseline information, (3) Confounders systematically identified, (4) New-user design used. **INVALID IF:** Assignment depends on post-baseline information. | `{assignment_variable, confounder_list, adjustment_method}` |
| **4. Outcomes** | Defines WHAT endpoints are measured. Must specify outcome definition, ascertainment method, timing, and validation status. | **VALID IF:** (1) Operationally defined with code algorithms, (2) Ascertainment begins AFTER time zero, (3) Validation status documented. **INVALID IF:** Outcomes vague or measured before time zero. | `{outcome_name, code_list, code_list_version, risk_window, induction_period, validation_status}` |
| **5. Start of Follow-up (Time Zero)** | Defines WHEN follow-up begins. The most critical alignment point. | **VALID IF:** Three-way coincidence: eligibility met = strategy assigned = outcomes counted. **INVALID IF:** Any misalignment exists. | `{index_date_definition, alignment_verified}` |
| **6. End of Follow-up** | Defines WHEN follow-up ends. Must specify censoring events and competing risks handling. | **VALID IF:** (1) Censoring events pre-specified, (2) Informative censoring addressed for per-protocol, (3) Competing risks handled. | `{max_followup, censoring_events[], competing_risks_method}` |
| **7. Causal Estimand** | Defines WHAT causal effect is estimated. Must specify ITT vs per-protocol and effect measure. | **VALID IF:** (1) Estimand explicit, (2) ICE handling defined, (3) Methods match estimand. **INVALID IF:** Ambiguous or per-protocol without g-methods plan. | `{estimand_type, effect_measure, ice_table[]}` |
| **8. Analysis Plan** | Defines HOW the effect will be estimated. Must be pre-specified. | **VALID IF:** (1) Pre-specified before data, (2) Methods appropriate for estimand, (3) Sensitivity analyses planned, (4) Diagnostics specified. | `{primary_method, sensitivity_analyses[], diagnostic_thresholds}` |

---

## Part II: Time-Zero Alignment (The Cardinal Rule)

> "Time zero of follow-up [is] the time when the eligibility criteria are met and a treatment strategy is assigned." — Hernán et al. 2016

### The Three-Way Coincidence Test

```
time_zero == eligibility_time == assignment_time == follow_up_start
```

**If this equation fails, the TTE is INVALID.**

### Hernán's Four Canonical Emulation Failures

| Failure | Description | Bias | Detection Rule |
|---------|-------------|------|----------------|
| **1. Time zero AFTER eligibility & assignment** | Left truncation; prevalent user inclusion | Selection bias | `IF treatment_start < time_zero THEN FAIL` |
| **2. Time zero at eligibility but AFTER assignment** | Post-treatment eligibility criteria | Selection bias | `IF eligibility_criteria uses post-assignment info THEN FAIL` |
| **3. Time zero BEFORE eligibility completion** | Eligibility requires sequential milestones but follow-up starts early | Immortal time | `IF eligibility requires future milestones THEN FAIL` |
| **4. Strategy assigned AFTER time zero** | Classic immortal time via future classification (e.g., "received ≥3 fills") | Immortal time | `IF exposure_definition_window_end > time_zero THEN FAIL` |

### Automated Validation Rules

```python
# Time-zero alignment checks (HARD FAIL)
assert eligibility_assessed_at == time_zero
assert treatment_assigned_at == time_zero
assert followup_starts_at == time_zero
assert all(outcome_dates > time_zero)

# No post-baseline eligibility (HARD FAIL)
for criterion in eligibility_criteria:
    assert criterion.evaluation_time <= time_zero
    assert "during follow-up" not in criterion.definition
    assert "must survive" not in criterion.definition

# No future exposure classification (HARD FAIL)
assert exposure_lookforward_window == 0 or design == "clone_censor_weight"
```

---

## Part III: Intercurrent Events (ICE) Table

Per ICH E9(R1), every TTE must define how intercurrent events are handled.

### Required ICE Table Structure

| Intercurrent Event | Handling Strategy | Implementation |
|--------------------|-------------------|----------------|
| Treatment discontinuation | treatment_policy / censoring / composite | Define deviation threshold |
| Treatment switching | treatment_policy / censoring | Define switch definition |
| Add-on therapy | treatment_policy / censoring | Define add-on definition |
| Death (non-outcome) | competing_risk / composite | Specify method |
| Loss to follow-up | censoring + IPW | Specify censoring model |

### ICE Handling Strategies

- **Treatment policy:** Include all events regardless of adherence (ITT-like)
- **Hypothetical:** Estimate effect if ICE had not occurred (requires modeling)
- **Composite:** ICE is part of outcome
- **While-on-treatment:** Censor at ICE (requires informative censoring adjustment)
- **Principal stratum:** Effect in subgroup that would not experience ICE

**Validation Rule:** `IF ice_handling == "censoring" THEN REQUIRE censoring_model_specified`

---

## Part IV: Bias Taxonomy

### A. Design-Induced Biases (TTE Prevents These)

| Bias | Structural Cause | Prevention | Detection |
|------|------------------|------------|-----------|
| **Immortal Time Bias** | Selection on post-assignment eligibility OR misclassification using future treatment | Time-zero alignment; clone-censor-weight | Exposure requires survival to qualify |
| **Lead Time Bias** | Earlier detection in one group | Align time zero identically | Outcome ascertainment asymmetric |
| **Prevalent User Bias** | Include patients already on treatment | New-user design | Treatment before time zero |
| **Depletion of Susceptibles** | Early events remove susceptible individuals | Incident user design | Time-varying HRs |

### B. Data-Driven Biases (Requires Adjustment)

| Bias | Mitigation | Required Check |
|------|------------|----------------|
| **Confounding by Indication** | PS methods, active comparator | DAG + balance diagnostics |
| **Unmeasured Confounding** | Negative controls, E-values | List unmeasured confounders |
| **Time-Varying Confounding** | G-methods (IPW, MSM, g-formula) | Required for per-protocol |
| **Measurement Error** | Validation, sensitivity analyses | Document validation status |
| **Informative Censoring** | IP weighting for censoring | Required for per-protocol |

### C. Time-Related Biases (Suissa 2020)

| Bias | Detection | Prevention |
|------|-----------|------------|
| **Protopathic Bias** | Spike in early event rates | Lag exposure; exclude early window |
| **Latency Time Bias** | Outcome has biological latency | Pre-specify induction period |
| **Time-Window Bias** | Asymmetric lookback windows | Enforce identical windows |
| **Immeasurable Time Bias** | Inpatient exposure not captured | Censor or bridge inpatient periods |

---

## Part V: Estimand Framework

> **Occam's Razor:** Choose the simplest estimand that answers the regulatory question.
> Use modern flexible methods (AIPW/SuperLearner) regardless of estimand complexity.

### Estimand Selection Guide

| Regulatory Question | Recommended Estimand | Rationale |
|---------------------|---------------------|-----------|
| "Is this drug effective?" | ITT Analogue | Answers policy question; real-world initiation effect |
| "Is this drug safe?" | ITT Analogue | Conservative for safety; captures intent-to-treat |
| "What if patients adhered?" | Per-Protocol | Requires adherence data + g-methods |
| "Dose-response effect?" | As-Treated | Requires time-varying dose data |

**Default to ITT Analogue** unless the causal question specifically requires adherence effects.

### Estimand Types

| Estimand | Definition | Methods | When to Use |
|----------|------------|---------|-------------|
| **ITT Analogue** ⭐ | Effect of initiating treatment | AIPW, PS methods | **Default for effectiveness/safety** |
| **Per-Protocol** | Effect of adhering to treatment | G-methods (IPW/MSM, g-formula) | When adherence effect is the question |
| **As-Treated** | Effect of actual treatment received | G-methods | When dose/duration matters |

> **Modern approach:** Use AIPW with SuperLearner (SOTAstack §3) for all estimands. It's doubly robust and handles high-dimensional confounding.

### Method Selection Logic

```
# For most effectiveness/safety trials:
IF estimand == "ITT_analogue":
    USE: AIPW with SuperLearner (SOTAstack §3)  # Modern, flexible
    REQUIRE: Baseline confounders only

# When adherence effect is specifically needed:
IF estimand == "per_protocol":
    USE: AIPW with censoring at deviation
    REQUIRE: Time-varying confounder adjustment
    REQUIRE: Censoring model + weight diagnostics
    WARN: More assumptions, more complexity
```

### Regulatory Interpretation Check

Before finalizing estimand, verify:
- [ ] The estimand answers the **actual regulatory question** (effectiveness OR safety)
- [ ] The interpretation is **clinically meaningful** (would change clinical practice)
- [ ] The simpler estimand (ITT) is **not sufficient** before choosing per-protocol
- [ ] Data supports the chosen estimand (adherence data available for per-protocol)

---

## Part VI: Design Patterns

> **For effectiveness/safety trials:** Use Pattern A (ACNU) with AIPW. Patterns B and C are rarely needed.

### A. Active Comparator New-User (ACNU) Design ⭐ RECOMMENDED

> **Code reference:** SOTAstack §1-3 implements this pattern

**Use When:** Comparing two active treatments (most effectiveness/safety questions)

**Requirements:**
- Both arms are new initiators (washout period specified)
- Time zero = first prescription/dispensation (`procedure_datetime`)
- Washout enforced: verify zero qualifying codes in lookback window
- Use AIPW with SuperLearner for estimation

**Code mapping:**
```python
# Index date (SOTAstack §1A)
index_date = procedure_datetime  # or drug_exposure_start_date

# Treatment assignment (SOTAstack §12)
T = procedure_concept_id IN CONTRAST_IDS  # vs NON_CONTRAST_IDS

# Baseline features: all OMOP domains BEFORE index_date (SOTAstack §1D)
# Outcome: condition_concept_id AFTER index_date within window (SOTAstack §5)
# Estimation: AIPW with SuperLearner (SOTAstack §3)
```

### B. Sequential Trial Emulation (Rarely Needed)

**Use When:** Comparing "initiate" vs "do not initiate" with no natural comparator

**Why rarely needed:** If an active comparator exists, use ACNU instead. Sequential trials are complex and require pooling with clustered variance.

**If needed:**
- Create trial at each eligible time point
- Non-initiators re-enter until initiation
- Pool with robust/clustered variance estimation

### C. Clone-Censor-Weight Design (Rarely Needed)

**Use When:** Grace periods, dynamic regimes, threshold-triggered treatment

**Why rarely needed:** For effectiveness/safety, index date is usually instant (first prescription). Grace periods add complexity without benefit.

**If needed:**
- Clone at time zero across compatible strategies
- Censor when incompatible with assigned strategy
- Apply IP weights for informative censoring

---

## Part VII: Diagnostic Thresholds

> **Code reference:** SOTAstack §4 (Robustness Checks), §7 (Important Metrics)

### Balance Diagnostics

| Metric | Target | Warning | Fail | Code |
|--------|--------|---------|------|------|
| **SMD (all covariates)** | < 0.1 | 0.1 - 0.25 | > 0.25 | `get_smd()` in §4B |
| **Variance Ratio** | 0.8 - 1.25 | 0.5 - 2.0 | < 0.5 or > 2.0 | — |

### Weighting Diagnostics

| Metric | Target | Warning | Action | Code |
|--------|--------|---------|--------|------|
| **Mean stabilized weight** | ≈ 1.0 | 0.9 - 1.1 | Check model | — |
| **Max weight** | < 10 | 10 - 20 | Truncate + sensitivity | — |
| **% Truncated** | < 1% | 1 - 5% | Report + sensitivity | — |
| **ESS / N** | > 0.5 | 0.3 - 0.5 | > 0.3 required | `calculate_ess()` in §4E |

### Positivity Diagnostics

| Check | Requirement | Code |
|-------|-------------|------|
| **PS trimming range** | [0.025, 0.975] | §9: PS_MIN, PS_MAX |
| **Overlap coefficient** | > 0.5 (good), > 0.3 (acceptable) | `calculate_overlap_coefficient()` in §4A |
| **PS AUC** | 0.6 - 0.85 optimal | — |
| **Common support %** | > 90% preferred | — |

### Effect Plausibility

| Check | Threshold | Action |
|-------|-----------|--------|
| **HR/RR implausibly protective** | < 0.3 for chronic outcome | Trigger time-bias audit |
| **E-value too low** | E-value < 1.5 | Weak against unmeasured confounding |
| **Negative control shows effect** | p < 0.05 | HIGH-SEVERITY BIAS ALERT |

> **Code reference:** `calculate_e_value()` in §4C, calibration in §4D

---

## Part VIII: Sensitivity Analyses (Required)

> **Code reference:** SOTAstack §4D (Empirical Null Calibration), §11 (Negative Controls)

| Analysis | Purpose | When Required | Code Reference |
|----------|---------|---------------|----------------|
| **Negative control outcomes** | Residual confounding detection | **Always** | §11: 65 pre-defined controls |
| **Empirical calibration** | Adjust estimates for systematic bias | **Always** | §4D: `calibrate_estimates()` |
| **E-value calculation** | Unmeasured confounding strength | **Always** | §4C: `calculate_e_value()` |
| **PS trimming sensitivity** | Positivity robustness | When using IPTW | §9: vary PS_MIN/PS_MAX |
| **Alternative exposure definitions** | Exposure misclassification | When codes ambiguous | — |
| **Alternative eligibility** | Population sensitivity | When judgment-based | — |
| **Confounder set variation** | Confounding adjustment | When key vars missing | — |
| **Subgroup analyses** | Effect heterogeneity | Pre-specified only | §6: ITE-based subgroups |

### Negative Control Implementation

```python
# SOTAstack provides 65 negative controls - use at least 10
# See §11 for full NEGATIVE_CONTROLS dictionary
# Run identical AIPW pipeline on each NC outcome
# Calibrate main results using empirical null (§4D)
```

**Minimum requirement:** Run ≥10 negative controls, calibrate P-values and CIs.

---

## Part IX: Decision Gates for Agents

### Gate 1: PICO Completeness (Pre-Flight)

```
REQUIRE: P (Population) → computable cohort definition
REQUIRE: I (Intervention) → well-defined strategy
REQUIRE: C (Comparator) → specified or defaultable
REQUIRE: O (Outcome) → measurable endpoint
IF any missing → STOP and request clarification
```

### Gate 2: Intervention Well-Definedness

```
CHECK: Can intervention be articulated as specific action?
CHECK: Would corresponding RCT be possible "in principle"?
IF vague (e.g., "loneliness", "obesity" without intervention) → STOP
```

### Gate 3: Data Mappability

```
FOR each protocol component:
    REQUIRE: data_mapping exists
    REQUIRE: code_list specified with version
    REQUIRE: time_window defined
IF any unmappable → Revise protocol or STOP
```

### Gate 4: Time-Zero Alignment (Critical)

```
VERIFY: eligibility_time == assignment_time == follow_up_start
VERIFY: exposure classification uses no future information
VERIFY: eligibility uses no post-baseline criteria
IF misalignment → HARD FAIL
```

### Gate 5: Estimand-Method Coherence

```
IF estimand == "per_protocol":
    REQUIRE: g-methods specified
    REQUIRE: time-varying confounders identified
    REQUIRE: censoring model specified
```

### Gate 6: Event Adequacy

```
CHECK: outcome_events >= 10 per covariate (rule of thumb)
IF rare outcome with complex model → WARN: consider hdPS or penalization
```

### Post-Estimation Gates

```
GATE 7: Effect Plausibility
    IF HR < 0.5 for non-acute outcome → TRIGGER time-bias audit

GATE 8: Negative Control Check
    IF negative_control shows effect → HIGH-SEVERITY BIAS ALERT

GATE 9: Balance Verification
    IF any SMD > 0.1 → WARN and document

GATE 10: Weight Diagnostics (IPW)
    IF max_weight > 20 → REQUIRE truncation + sensitivity
    IF ESS/N < 0.2 → FAIL: insufficient effective sample
```

---

## Part X: Edge Cases and Special Scenarios

### Common Pitfalls to Handle

| Scenario | Issue | Solution |
|----------|-------|----------|
| **Switching/add-on therapy** | Is it ICE, confounder, or part of regime? | Define explicitly in ICE table |
| **Stockpiling/early refills** | Overlapping fills inflate exposure | Use days' supply with overlap rules |
| **In-hospital exposure** | Inpatient meds not in claims | Censor inpatient periods or flag |
| **Informative encounters** | "Must have lab" creates selection | Don't condition on post-baseline labs |
| **Competing risks dominating** | Death before outcome (elderly) | Explicit estimand (cause-specific vs CIF) |
| **Calendar time confounding** | Guideline changes, COVID shifts | Stratify by calendar period |
| **Multiple indications** | Drug used for different diseases | Restrict by indication + active comparator |
| **Confirmation visit requirement** | "2 fills" or "return visit" | Immortal time risk - use clone-censor-weight |

---

## Part XI: Reporting Requirements (TARGET Standards)

### Required Documentation

1. **Study identification:** Explicitly identify as target trial emulation
2. **Causal question:** State question and reason for TTE approach
3. **Target trial protocol:** Full specification (all 8 components + ICE table)
4. **Emulation mapping:** How each component maps to data
5. **Identifying assumptions:** Exchangeability, positivity, consistency
6. **Estimand precision:** Point estimate + confidence intervals
7. **Sensitivity analyses:** All planned analyses with results
8. **Limitations:** Explicit threats to validity

### Required Artifacts

- [ ] Target trial protocol table
- [ ] Emulation mapping table
- [ ] DAG (directed acyclic graph)
- [ ] Cohort attrition diagram
- [ ] Baseline characteristics table (by arm)
- [ ] Balance table (SMDs before/after)
- [ ] PS distribution plots (if applicable)
- [ ] Weight diagnostics (if IPW)
- [ ] Outcome curves (KM or cumulative incidence)
- [ ] Forest plot (sensitivity analyses)

---

## Part XII: Quick Reference Checklists

### Protocol Completeness

- [ ] Eligibility criteria (baseline only, with code lists)
- [ ] Treatment strategies (with grace period handling)
- [ ] Assignment procedure (new-user, time-zero aligned)
- [ ] Outcomes (with code lists, validation status)
- [ ] Follow-up start (= eligibility = assignment)
- [ ] Follow-up end (censoring rules, competing risks)
- [ ] Estimand (ITT/per-protocol, ICE table)
- [ ] Analysis plan (methods, diagnostics, sensitivity)

### Immortal Time Prevention

- [ ] No post-baseline eligibility criteria
- [ ] No future-based exposure classification
- [ ] Three-way coincidence verified
- [ ] Grace periods use clone-censor-weight
- [ ] No "must survive X" or "must receive X doses"

### Confounding Control

- [ ] Confounders identified (literature + DAG)
- [ ] All key confounders measurable
- [ ] Unmeasured confounders listed
- [ ] Method appropriate for estimand
- [ ] Balance diagnostics planned (SMD < 0.1)
- [ ] Positivity assessed

### Data Quality

- [ ] Code lists versioned
- [ ] Lookback windows specified
- [ ] Washout enforced and verified
- [ ] Continuous enrollment verified
- [ ] Outcome validation status documented

---

## Appendix: Validation Rule Summary (Machine-Checkable)

```yaml
# Protocol Completeness
- rule: all_protocol_fields_present
  check: eligibility AND strategies AND assignment AND outcomes AND time_zero AND followup_end AND estimand AND sap
  severity: HARD_FAIL

# Time-Zero Alignment
- rule: three_way_coincidence
  check: eligibility_time == assignment_time == followup_start
  severity: HARD_FAIL

- rule: no_postbaseline_eligibility
  check: all eligibility criteria evaluate at <= time_zero
  severity: HARD_FAIL

- rule: no_future_exposure_classification
  check: exposure_lookforward == 0 OR design == clone_censor_weight
  severity: HARD_FAIL

# Estimand-Method Coherence
- rule: perprotocol_requires_gmethods
  check: IF estimand == per_protocol THEN method IN [ipw_msm, gformula, gestimation]
  severity: HARD_FAIL

- rule: perprotocol_requires_censoring_model
  check: IF estimand == per_protocol THEN censoring_model_specified
  severity: HARD_FAIL

# ICE Handling
- rule: ice_table_complete
  check: all intercurrent_events have handling_strategy
  severity: WARN

- rule: censoring_ice_requires_model
  check: IF ice_handling == censoring THEN censoring_model_specified
  severity: HARD_FAIL

# Diagnostics
- rule: smd_threshold
  check: all SMD < 0.25
  severity: WARN if > 0.1, FAIL if > 0.25

- rule: ess_threshold
  check: ESS/N > 0.2
  severity: HARD_FAIL

- rule: weight_truncation
  check: IF max_weight > 20 THEN truncation_applied
  severity: WARN

# Effect Plausibility
- rule: implausible_effect
  check: IF HR < 0.3 for chronic outcome THEN trigger_bias_audit
  severity: WARN
```

---

## Appendix B: Spec → Code Translation Guide

This appendix maps rubric concepts to SOTAstack implementation.

### Protocol Component → SQL/Python

| Rubric Component | SOTAstack Section | Key Code |
|------------------|-------------------|----------|
| Eligibility criteria | §1A, §1D | `WHERE procedure_concept_id IN (...)` |
| Treatment strategies | §12 | `CONTRAST_IDS`, `NON_CONTRAST_IDS` |
| Time zero | §1A | `CAST(procedure_datetime AS DATE) AS index_date` |
| Outcomes | §5 | `ANALYSIS_OUTCOMES` dict with concept IDs + windows |
| Baseline confounders | §1D, §8 | High-dim features from all OMOP domains `< index_date` |
| Propensity score | §2 | `LogisticRegression(penalty='l1', solver='saga', C=0.2)` |
| Effect estimation | §3 | `run_cross_fitted_aipw()` |
| Balance check | §4B | `get_smd()` with sparse matrix support |
| Sensitivity | §4D, §11 | `calibrate_estimates()` + `NEGATIVE_CONTROLS` |

### Diagnostic Threshold Alignment

| Rubric Threshold | SOTAstack Threshold | Code Location |
|------------------|---------------------|---------------|
| SMD < 0.1 | SMD < 0.1 | §4B, §7 |
| ESS/N > 0.2 | ESS/N > 0.5 (stricter) | §4E, §7 |
| Max weight < 10 | Implicit via PS trim | §9 |
| PS range | [0.025, 0.975] | §9: `PS_MIN`, `PS_MAX` |
| Min events | 10 per group | §5, §6 |
| Min subgroup N | 100-150 | §6 |
| Min negative controls | 10 for calibration | §4D |

### Estimand → Method Mapping

| Estimand | SOTAstack Implementation |
|----------|-------------------------|
| ITT Analogue | §3: AIPW with SuperLearner (default) |
| Risk Difference (ATE) | §3: `stats['ATE']` from EIF |
| Risk Ratio | §3: `stats['RR']` = Risk_1 / Risk_0 |
| Hazard Ratio | §10: Cox with IPTW weights |
| Subgroup effects | §6: ITE-based, no refitting |

### Output Artifacts

| Rubric Requirement | SOTAstack Output |
|--------------------|------------------|
| Baseline table | `df_cohort` with demographics |
| Balance table | SMD before/after from `get_smd()` |
| PS distributions | `ps_control`, `ps_treated` histograms |
| Weight diagnostics | ESS, overlap coefficient |
| Outcome curves | KM with IPTW weights (§10) |
| Effect estimates | `stats` dict from AIPW |
| Calibrated estimates | `calibrated_results` from §4D |

---

*This rubric is derived from: Hernán et al. 2016 (AJE, J Clin Epidemiol), Hernán et al. 2022 (JAMA), Hernán et al. 2025 (Ann Intern Med, Epidemiology), Matthews et al. 2022 (BMJ), NICE RWE Framework 2022, EMA CHMP/MWP 2025, ICH E9(R1), Zuo et al. 2023, Suissa & Dell'Aniello 2020, Cain et al. 2010, Danaei et al. 2013, Lipsitch et al. 2010, Fu 2023 (JASN), TARGET statement 2025, and RCT-DUPLICATE initiative.*

*Code implementation reference: SOTAstack.md (CausalContrastStudy methodology)*
