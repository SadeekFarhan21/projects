# Target Trial Emulation (TTE) Protocol Specification  
**Causal question:** Among adults with CKD (baseline eGFR <60 mL/min/1.73m²) undergoing **IV contrast-enhanced multidetector CT (MDCT)**, what is the effect of receiving **iodixanol (iso-osmolar)** vs **iopamidol (low-osmolar)** on **72-hour risk of CA-AKI**, defined as **increase in serum creatinine ≥0.5 mg/dL within 72 hours** after contrast administration?

**Design pattern (default):** Active Comparator New-User (ACNU) at the *scan-level* + cross-fitted AIPW (doubly robust) with high-dimensional baseline covariates.

**Data model assumption:** OMOP CDM v5.3+ with timestamps available for drug/procedure/measurement records.  
**Unit of analysis:** Primary = **first eligible contrast-enhanced MDCT per person** (to avoid within-person correlation). Sensitivity = all eligible scans with person-clustered SE.

---

## 0) Target Trial Protocol (Hernán & Robins “7 components”)

| Component | Target trial (ideal RCT) | Emulation in OMOP EHR/claims |
|---|---|---|
| Eligibility | Adults with CKD (eGFR<60) receiving IV contrast MDCT | Baseline labs + CT procedure + contrast agent exposure in OMOP with pre-index lookback |
| Strategies | Administer iodixanol vs iopamidol for the MDCT | Contrast agent identified from `drug_exposure` (preferred) and/or `procedure_occurrence` add-ons |
| Assignment | Randomize at time of contrast | Observed assignment; adjust confounding via LSPS + AIPW |
| Outcome | Creatinine rise ≥0.5 mg/dL within 72h | Lab-based algorithm using `measurement` creatinine values |
| Time zero | Contrast administration datetime | **Scan episode anchor datetime (`t0`) defined from linked CT + contrast records (see §5)** |
| Follow-up | 0 to 72 hours | Administrative end at +72h; handle death/dialysis/measurement-missingness explicitly |
| Estimand | ITT-like initiation effect | ATE on 72h risk (RD primary; RR secondary) |
| Analysis | Compare risks at 72h | Cross-fitted AIPW (+ monitoring/competing-risk handling + diagnostics + calibration) |

---

## 1) Eligibility Criteria (Component 1)

### 1.1 Inclusion / Exclusion Table (OMOP-mappable; baseline-only)

**Key rule:** *No eligibility criterion may require post–time-zero information* (e.g., “must have follow-up creatinine”). Missing post-contrast labs will be handled analytically (see §8.3).

> **Notation:** `t0` = time zero (scan episode anchor datetime; see §5).

| # | Criterion | Type | Data mapping (OMOP) | Lookback / assessment window | Code list | Code list version |
|---:|---|---|---|---|---|---|
| I1 | Age ≥18 at `t0` | Include | `person.birth_datetime` + `t0` | At `t0` | — | — |
| I2 | Contrast-enhanced MDCT performed at index | Include | Define a **scan episode** by linking a CT/MDCT-with-IV-contrast `procedure_occurrence` to an iodinated contrast `drug_exposure` (I3/I4) **within the same visit**. Set `t0` per §5. | **Linking window:** absolute time difference ≤6h (datetime-aware); if only dates, require same calendar date. | `[OMOP: contrast-enhanced CT/MDCT procedure concepts]` | local / Athena versioned concept set |
| I3 | Received **iodixanol** IV for the index scan (intervention arm) **OR** | Include | `drug_exposure` (preferred): `drug_concept_id` for iodixanol; route = IV (if available); exposure record must be part of the linked scan episode used to define `t0` | At `t0` (assignment at scan episode anchor) | `[OMOP: iodixanol]` | RxNorm / local mapping snapshot |
| I4 | Received **iopamidol** IV for the index scan (comparator arm) | Include | `drug_exposure`: iopamidol; part of linked scan episode | At `t0` (assignment at scan episode anchor) | `[OMOP: iopamidol]` | RxNorm / local mapping snapshot |
| I5 | Baseline serum creatinine available pre-contrast | Include | `measurement` creatinine; choose “most recent qualifying” value before `t0` | **Primary (datetime available):** within 24h pre-`t0` if inpatient/ED; within 7d pre-`t0` if outpatient (visit-type stratified rule). **Date-only tie-break:** require `measurement_date < t0_date` (see §1.2 / §5.2). | `[OMOP: serum creatinine measurement]` | LOINC/OMOP Measurement |
| I6 | Baseline eGFR <60 mL/min/1.73m² | Include | Prefer `measurement` eGFR; else compute using CKD-EPI (race-free 2021 recommended) from creatinine + age + sex | Same window as baseline creatinine | `[OMOP: eGFR measurement]` (optional) | LOINC/OMOP Measurement |
| I7 | Sufficient baseline history for confounding control | Include | `observation_period` contains continuous observation | ≥180 days pre-`t0` | — | — |
| E1 | Maintenance dialysis / ESRD before `t0` | Exclude | `procedure_occurrence` dialysis OR `condition_occurrence` ESRD OR `device_exposure` dialysis access (optional) | 365d pre-`t0` (and/or anytime prior) | `[OMOP: chronic hemodialysis]`, `[OMOP: peritoneal dialysis]`, `[OMOP: ESRD]` | versioned concept sets |
| E2 | Kidney transplant before `t0` (optional but common) | Exclude | `procedure_occurrence` transplant OR `condition_occurrence` transplant status | Anytime prior (or 5y) | `[OMOP: kidney transplant]` | versioned concept set |
| E3 | Evidence of **AKI at baseline** (to reduce reverse causation) | Exclude | `condition_occurrence` AKI diagnosis **or** lab-based AKI rule if feasible | 7d pre-`t0` for diagnosis; lab-based rule uses prior creatinine if available | `[OMOP: acute kidney injury]` + creatinine | versioned |
| E4 | **Recent iodinated contrast exposure** (washout) | Exclude | `drug_exposure` iodinated contrast media OR `procedure_occurrence` contrast administration | **Primary:** 7d pre-`t0`; sensitivity: 30d | `[OMOP: iodinated contrast media]` | versioned |
| E5 | Multiple iodinated contrast agents **concurrently at assignment** (mixed exposure at baseline) | Exclude | Two+ distinct iodinated contrast `drug_concept_id`s that are **near-simultaneous** and can be treated as part of assignment | **Datetime-aware:** both start times in **[t0−15min, t0]** (or same timestamp). **Date-only:** two+ distinct agents recorded on `t0_date` in the same visit. | `[OMOP: iodinated contrast media]` | versioned |
| E6 | Intra-arterial contrast procedure **at/before `t0`** (route-mixing) | Exclude | `procedure_occurrence` angiography/cath with contrast **with procedure_datetime ≤ t0** (or date-only: procedure_date ≤ t0_date) | **Primary:** within **[-1d, 0]** relative to `t0` (baseline-only) | `[OMOP: angiography with contrast]` | versioned |
| E7 | Missing sex (if needed for eGFR calculation) | Exclude | `person.gender_concept_id` missing | At `t0` | — | — |

### 1.2 Baseline creatinine and eGFR operationalization (pre-specified)
- **Baseline creatinine (`Cr0`) selection rule:** most recent serum creatinine **strictly before** `t0` within the window determined by visit setting:
  - If `visit_occurrence.visit_concept_id` indicates inpatient/ED: choose value in **(t0−24h, t0)**.
  - If outpatient: choose value in **(t0−7d, t0)**.
- **Date-only tie-break rule (when datetimes are unavailable for `t0` and/or creatinine):**
  - Define `t0_date = DATE(t0)` (from the scan episode anchor date).
  - Accept baseline creatinine only if `measurement_date < t0_date` (i.e., exclude same-day date-only creatinine as ambiguous for strict pre/post ordering).
  - For baseline windows, implement the lookback in whole days (e.g., inpatient/ED: `measurement_date ∈ {t0_date−1}`; outpatient: `measurement_date ∈ [t0_date−7, t0_date−1]`).
- **Baseline eGFR (`eGFR0`) rule:**
  1) If an eGFR measurement exists in the same baseline window, use the most recent eGFR value; else  
  2) Compute eGFR using CKD-EPI 2021 race-free equation from `Cr0`, age, and sex.
- **CKD criterion:** `eGFR0 < 60`.

---

## 2) Treatment Strategies (Component 2)

### 2.1 Strategies (well-defined at time zero)

| Arm | Strategy | Initiation rule | Grace period | Adherence definition | Allowed switching / add-ons | Deviation rules |
|---|---|---|---:|---|---|---|
| A (Intervention) | **Iodixanol** IV administered for index MDCT | `drug_exposure` contains iodixanol as part of the linked scan episode defining `t0` | None | Single administration at `t0` | Usual care after scan allowed (IV fluids, meds) | Exclude if concurrent mixed iodinated agents at assignment (E5) |
| B (Comparator) | **Iopamidol** IV administered for index MDCT | `drug_exposure` contains iopamidol as part of the linked scan episode defining `t0` | None | Single administration at `t0` | Same | Same |

**Dose/volume constraints (pragmatic):**
- Do **not** restrict dose/volume in primary analysis unless reliably captured.
- If contrast volume is captured, include it as a baseline covariate (measured at/just before `t0`) and run a sensitivity analysis restricting to typical volume range (site-specific).

### 2.2 Exposure ascertainment (linking rules)
To reduce mislinking of contrast to other encounters:

- Construct a **scan episode** by requiring:
  1) A qualifying MDCT-with-IV-contrast `procedure_occurrence`, and  
  2) A qualifying iodinated contrast `drug_exposure` (iodixanol or iopamidol),  
  in the **same `visit_occurrence_id`** (preferred) and temporally proximate.

- **Temporal proximity rule (datetime-aware):**
  - Require `|procedure_datetime − drug_exposure_start_datetime| ≤ 6 hours`.
  - Define `t0` as the **scan episode anchor datetime** (see §5): `t0 = max(procedure_datetime, drug_exposure_start_datetime)` so that both eligibility evidence and assignment evidence occur **at or before** `t0`.

- **If only dates are available (date-only mode):**
  - Require `procedure_date == drug_exposure_start_date` (same calendar date) within the same visit.
  - Define `t0_date` as that shared date; set `t0` to `t0_date` (date) for windowing, using the tie-break rules in §4.1 and §5.2.

---

## 3) Treatment Assignment / Emulation (Component 3)

### 3.1 Assignment variable (observational)
- `T = 1` if iodixanol at `t0`, `T = 0` if iopamidol at `t0`.

### 3.2 New-user / washout specification (scan-level)
Because contrast is an acute exposure, “new-user” is interpreted as **no recent iodinated contrast**:
- **Primary washout:** no iodinated contrast `drug_exposure` in **7 days** prior to `t0`.
- **Sensitivity washout:** 30 days.
- **Primary analysis uses first eligible scan per person** after satisfying washout.

### 3.3 Confounder set (baseline-only)
You will adjust using:
1) **Clinically forced covariates** (always included, even in high-dim modeling), and  
2) **High-dimensional OMOP features** from all domains in a pre-index lookback window.

**Clinically forced baseline covariates (minimum set):**
- Demographics: age, sex, calendar time (year/quarter), site/facility (if available)
- Renal function: `Cr0`, `eGFR0`, CKD stage category (e.g., 3a/3b/4/5), proteinuria if available
- Comorbidities (lookback 365d unless noted): diabetes, hypertension, heart failure, CAD, PVD, stroke, COPD, cirrhosis, malignancy
- Acute illness/severity at `t0`: inpatient vs ED vs outpatient, ICU on index day, sepsis/shock/hypotension codes, recent surgery/trauma (30d)
- Nephrotoxic co-medications (active in 7–30d pre-`t0`): NSAIDs, ACEi/ARB, diuretics, aminoglycosides, vancomycin, amphotericin, calcineurin inhibitors, metformin (as marker), chemotherapy (broad)
- Imaging/contrast context: CT body region/type, contrast volume (if available), repeat CT orders (baseline), prophylaxis planned at/before `t0` (IV isotonic fluids; sodium bicarb; NAC) if captured
- Prior AKI history: any AKI diagnosis in 365d (excluding baseline AKI exclusion window)

**High-dimensional baseline features (hdPS-style in OMOP):**
- All `condition_occurrence`, `procedure_occurrence`, `drug_exposure`, `measurement` (presence indicators), `observation`, `device_exposure` features in **[-180d, -1h]** relative to `t0`, prevalence-filtered (e.g., ≥50 persons).

---

## 4) Outcomes (Component 4)

### 4.1 Primary outcome (lab-based; matches question)
**Outcome name:** CA-AKI (classic creatinine criterion) within 72 hours  
**Definition:**  
Let `Cr0` = baseline creatinine (pre-contrast) and `Crmax72` = maximum post-contrast creatinine measured within (t0, t0+72h].  
- **Event** if `Crmax72 − Cr0 ≥ 0.5 mg/dL`.

**OMOP operationalization:**
- Baseline and follow-up creatinine from `measurement` where `measurement_concept_id IN [OMOP: serum creatinine measurement]`.
- Compute `Crmax72` using `measurement_datetime` (or `measurement_date` if datetime unavailable) constrained to window.
- Use **peak** in-window value.

**Datetime vs date-only handling (deterministic tie-break; required for strict pre/post ordering):**
- **Preferred (datetime available):**
  - Baseline must satisfy `Cr0_time < t0`
  - Follow-up creatinine must satisfy `t0 < Cr_time ≤ t0+72h`
- **If only dates are available for `t0` and/or creatinine:**
  - Define `t0_date = DATE(t0)`.
  - **Baseline (`Cr0`) eligible only if** `measurement_date < t0_date` (exclude same-day date-only values as ambiguous).
  - **Follow-up (`Crmax72`) eligible only if** `measurement_date > t0_date` and `measurement_date ≤ t0_date + 3 days` (approximating 72 hours without within-day ordering).
  - Report the proportion of episodes using datetime vs date-only (`t0` datetime present; baseline creatinine datetime present; follow-up creatinine datetime present).

**Risk window:** 0 to 72 hours after `t0`  
**Induction period:** 0 hours (sensitivity: start at 24h to reduce immediate peri-procedural effects)

**Validation status:** Lab-based algorithm; face-valid but site-specific capture and timing may vary (document local measurement completeness).

### 4.2 Key secondary outcomes (recommended)
1) **KDIGO-aligned CA-AKI** within 48–72h (depending on available timing):  
   - `Cr increase ≥0.3 mg/dL within 48h` OR `≥1.5× baseline within 7 days` (operationalize per available window; pre-specify exact window).  
2) **Acute dialysis within 72h** (procedure-based)  
3) **Composite**: CA-AKI (primary) **or death** within 72h (to address competing risk in a patient-important way)

### 4.3 Outcome coding fields (rubric-required)
| Outcome | OMOP domain | Code list | Code list version | Window | Notes |
|---|---|---|---|---|---|
| Primary CA-AKI (Cr rise ≥0.5) | Measurement-derived | `[OMOP: serum creatinine measurement]` | versioned | 72h | Computed, not a condition code |
| Acute dialysis | Procedure | `[OMOP: hemodialysis procedure]`, `[OMOP: continuous renal replacement therapy]` | versioned | 72h | Use for competing risk / composite |
| Death | Person | `person.death_date` / death table | — | 72h | Competing event + composite |

---

## 5) Time Zero (Component 5)

### 5.1 Index date/time definition
**Time zero (`t0`) = the scan episode anchor datetime for the linked MDCT + contrast administration episode**, defined to preserve three-way coincidence (eligibility met = strategy assigned = outcomes counted) without relying on post–time-zero eligibility evidence:

1) Identify a linked **scan episode** consisting of:
   - an MDCT-with-IV-contrast `procedure_occurrence` record, and
   - an iodixanol or iopamidol `drug_exposure` record,
   within the same visit and within the temporal proximity rule in §2.2.

2) **Define `t0` as:**
   - **Primary (datetime-aware):** `t0 = max(procedure_datetime, drug_exposure_start_datetime)`  
     (so both the qualifying CT and the contrast assignment record occur **at or before** `t0`)
   - **Date-only mode:** if only dates are available, define `t0_date` as the shared calendar date (per §2.2) and use `t0_date` for windowing (see §4.1 and §5.2).

### 5.2 Three-way coincidence verification (HARD validity gate)
You must be able to assert (programmatically) that:
- Eligibility is assessed using only data **≤ t0** (baseline creatinine strictly pre-`t0` under the applicable datetime/date-only rule).
- Treatment assignment (iodixanol vs iopamidol) is determined **at t0** (by the linked contrast record in the scan episode).
- Outcome counting begins **after t0** (post-contrast creatinine measurements strictly after `t0`, or strictly after `t0_date` in date-only mode).

**Machine-check rules (datetime-aware):**
- `Cr0_time < t0`
- all post-contrast creatinine times used for `Crmax72` satisfy `t0 < Cr_time ≤ t0+72h`
- scan episode linkage: `abs(procedure_datetime - drug_exposure_start_datetime) ≤ 6h` and same visit (preferred)

**Machine-check rules (date-only mode):**
- Define `t0_date = DATE(t0)` (or the scan episode date)
- `Cr0_date < t0_date`
- all post-contrast creatinine dates used satisfy `t0_date < Cr_date ≤ t0_date + 3 days`
- linkage requires `procedure_date == drug_exposure_start_date` (same visit preferred)

---

## 6) Follow-up (Component 6)

### 6.1 Follow-up period
- Start: `t0`
- End: earliest of
  - `t0 + 72 hours` (administrative end)
  - end of `observation_period` (loss of data capture)
  - death (competing event)
  - acute dialysis initiation (competing event; also secondary endpoint)

### 6.2 Censoring and competing risks (pre-specified)
- **Death within 72h:** treat as a **competing event** for lab-defined CA-AKI; also analyze a **composite (CA-AKI or death)**.
- **Acute dialysis within 72h:** treat as competing event and as secondary endpoint; optionally include in composite (CA-AKI or dialysis or death).

### 6.3 Missing post-contrast creatinine measurements
Do **not** exclude at baseline for missing future labs. Handle analytically (see §8.3).

---

## 7) Causal Estimand + ICE Table (Component 7)

### 7.1 Primary estimand (ITT analogue; initiation effect)
**Estimand type:** ITT analogue (treatment-policy)  
**Population:** Eligible adults with CKD undergoing contrast-enhanced MDCT at `t0`  
**Treatment comparison:** iodixanol vs iopamidol at `t0`  
**Outcome:** CA-AKI event by 72h (Cr rise ≥0.5)  
**Effect measures:**
- **Primary:** Risk Difference (ATE on 72h cumulative incidence)
- **Secondary:** Risk Ratio (RR) at 72h

### 7.2 Intercurrent Events (ICE) Table (ICH E9[R1]-style)

| Intercurrent event (0–72h) | Strategy | Implementation in emulation |
|---|---|---|
| Additional iodinated contrast (repeat imaging) | Treatment policy (primary); sensitivity censor | Primary: ignore (part of real-world strategy). Sensitivity: censor at time of additional contrast; apply IPCW for censoring if done. |
| Post-contrast prophylactic IV fluids given after t0 | Treatment policy | Do not censor; adjust if measured at/near baseline; otherwise captured indirectly via hd features. |
| Nephrotoxic meds started after t0 | Treatment policy | Do not censor (post-baseline mediator/ICE). |
| Death before 72h | Competing risk + composite secondary | Primary endpoint: competing risk acknowledged; report composite (CA-AKI or death) as key secondary. |
| Acute dialysis before 72h | Competing risk + secondary | Competing risk; also separate and composite endpoints. |
| Loss of observation / missing labs | Censoring + IPCW (monitoring model) | Model probability of having outcome ascertainment (creatinine measured) and/or remaining observable through 72h. |

---

## 8) Analysis Plan (Component 8)

### 8.1 Primary analysis approach (ACNU + cross-fitted AIPW)
1) **Extract cohort** per eligibility and exposure definitions.
2) **Create baseline covariate matrix**:
   - Dense forced covariates (age, sex, `Cr0`, `eGFR0`, visit type, etc.) + missingness indicators
   - Sparse high-dimensional OMOP features from all domains in [-180d, -1h]
3) **Estimate nuisance functions with cross-fitting (K=5 folds):**
   - Propensity score: \(\pi(X)=P(T=1\mid X)\) using SuperLearner (lasso + RF) or LSPS lasso
   - Outcome regressions: \(\mu_t(X)=E(Y\mid T=t,X)\) for t∈{0,1}
4) **Trim propensity scores** to [0.025, 0.975] and compute diagnostics.
5) **Estimate ATE (risk difference) using AIPW EIF**; report RD, RR, SE, 95% CI, p-value.

### 8.2 Diagnostics (must be reported; rubric thresholds)
- **Covariate balance:** absolute SMD <0.1 target; warn if any >0.1; fail if any >0.25
- **Positivity:** overlap coefficient >0.5 preferred (>0.3 acceptable); report PS histograms
- **Weight stability (if using IPTW diagnostics):** max weight <10 target; ESS/N >0.5 target; fail if ESS/N ≤0.2
- **Event adequacy:** ensure ≥10 outcome events per arm before complex nuisance models; otherwise simplify (stronger regularization / fewer learners)

### 8.3 Primary handling of missing post-contrast creatinine (monitoring/ascertainment)
Because CA-AKI is lab-defined, outcome may be **missing** if no creatinine is measured within 72h.

**Primary approach (recommended): AIPW with an outcome-ascertainment model**
- Define \(R=1\) if **any** creatinine measurement exists in (t0, t0+72h], else \(R=0\).
- Model \(g(T,X)=P(R=1\mid T,X)\) using SuperLearner (baseline-only predictors).
- Use a **doubly robust missing-outcome AIPW** estimator (pre-specify in SAP) where augmentation terms are weighted by \(1/g\).  
  This targets the 72h risk under the assumption that (conditional on T and X) lab ascertainment is “missing at random”.

**Required reporting:**
- Proportion with \(R=1\) by arm
- Distribution of \(g(T,X)\) and any truncation (e.g., truncate g to [0.05, 1])

**Sensitivity analyses for missingness:**
- Complete-case analysis (R=1 only) **clearly labeled as biased-prone**
- Worst/best-case bounds (e.g., assume all missing in one arm had event vs none) to bracket plausible effects
- Restrict to inpatient/ED scans (baseline-defined) where monitoring is more protocolized

### 8.4 Competing risk analyses (72h horizon)
Report, at minimum:
- Primary CA-AKI (lab-defined) with competing events described
- **Composite endpoint:** CA-AKI **or death** by 72h (patient-important, avoids “informative survival”)
- Sensitivity: treat death as censoring with IPCW vs treat as composite (compare)

### 8.5 Sensitivity analyses (required)
1) **Alternative outcome definitions**
   - KDIGO CA-AKI (≥0.3 mg/dL within 48h; ≥1.5× baseline within 7d if feasible)
   - Alternative post window: 24–72h (reduce immediate peri-procedural effects)
2) **Alternative baseline definition**
   - Baseline creatinine within 48h inpatient / 14d outpatient
3) **Alternative washout**
   - 30-day no prior iodinated contrast (vs 7-day primary)
4) **Exposure definition robustness**
   - Require explicit IV route recorded (if available) vs not required
   - Restrict to common concentrations/formulations (site-specific)
5) **Handling repeat scans**
   - Use all eligible scans with person-clustered SE vs first-scan-only
6) **Propensity trimming sensitivity**
   - Compare [0.01,0.99] vs [0.025,0.975]

### 8.6 Negative controls + empirical calibration (required, with feasibility note)
For a **72-hour** window, many “classic” negative control outcomes may have too few events. Pre-specify:
- Run ≥10 negative control outcomes using a **longer window (e.g., 30 days)** to ensure non-zero events, keeping the *same exposure, baseline covariates, and estimation pipeline*.  
- Use these to fit an empirical null and calibrate p-values/CIs for the primary endpoint (report both uncalibrated and calibrated).

### 8.7 Subgroup analyses (pre-specified only; no refitting)
Use ITE-based subgroup summaries (no nuisance refit) for:
- Baseline eGFR strata: <30; 30–44; 45–59
- Diabetes (yes/no)
- Inpatient/ED vs outpatient
Minimum N per subgroup: pre-specify (e.g., ≥150 total and ≥5 events/arm).

---

## 9) Emulation Mapping to OMOP (implementation-ready summary)

| Protocol element | OMOP tables | Key fields | Notes |
|---|---|---|---|
| Index MDCT | `procedure_occurrence` | `procedure_concept_id`, `procedure_datetime`, `visit_occurrence_id` | Use concept set for contrast-enhanced CT/MDCT |
| Contrast agent | `drug_exposure` | `drug_concept_id`, `drug_exposure_start_datetime`, `route_concept_id`, `dose_unit_concept_id`, `quantity` | Prefer drug_exposure; if absent, use local procedure/charge mapping |
| Baseline creatinine/eGFR | `measurement` (+ `person`) | `measurement_concept_id`, `value_as_number`, `measurement_datetime` | Compute eGFR if needed |
| Comorbidities | `condition_occurrence` | `condition_concept_id`, `condition_start_date` | Lookback 365d (typical) |
| Concomitant meds | `drug_exposure` | drug codes + dates | Lookback 30–180d depending on med |
| Death | `person` / death table | `death_date` | Use within 72h for competing/composite |
| Dialysis | `procedure_occurrence` | dialysis procedure concepts | Competing/secondary |

---

## OMOP_LOOKUP_TERMS
iodixanol|Drug  
iopamidol|Drug  
iodinated contrast media|Drug  
computed tomography with intravenous contrast|Procedure  
multidetector computed tomography|Procedure  
CT angiography with contrast|Procedure  
angiography with contrast (intra-arterial)|Procedure  
serum creatinine|Measurement  
estimated glomerular filtration rate (eGFR)|Measurement  
blood urea nitrogen (BUN)|Measurement  
acute kidney injury|Condition  
chronic kidney disease|Condition  
end-stage renal disease|Condition  
hemodialysis (maintenance)|Procedure  
peritoneal dialysis (maintenance)|Procedure  
continuous renal replacement therapy (CRRT)|Procedure  
kidney transplant|Procedure  
kidney transplant status|Condition  
diabetes mellitus type 2|Condition  
hypertension|Condition  
heart failure|Condition  
coronary artery disease|Condition  
peripheral arterial disease|Condition  
stroke|Condition  
chronic obstructive pulmonary disease|Condition  
cirrhosis|Condition  
malignancy (any cancer)|Condition  
sepsis|Condition  
shock|Condition  
hypotension|Condition  
dehydration|Condition  
intensive care unit admission|Procedure  
nonsteroidal anti-inflammatory drugs (NSAIDs)|Drug  
ACE inhibitor|Drug  
angiotensin receptor blocker|Drug  
loop diuretic|Drug  
thiazide diuretic|Drug  
metformin|Drug  
vancomycin|Drug  
aminoglycosides|Drug  
amphotericin B|Drug  
calcineurin inhibitor (tacrolimus/cyclosporine)|Drug  
chemotherapy (systemic antineoplastic agents)|Drug  
intravenous normal saline hydration|Drug  
sodium bicarbonate infusion|Drug  
N-acetylcysteine|Drug  
pregnancy|Condition