import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & OMOP CONCEPT IDS
# =============================================================================

# Validated Concept IDs (from Prompt)
# -----------------------------------
# Exposures (Specific Contrast Agents for VALOR Trial)
IODIXANOL_CONCEPT_ID = 19003201  # Iso-osmolar (Intervention)
IOVERSOL_CONCEPT_ID = 19069131   # Low-osmolar (Comparator)

# Procedure: Coronary Angiography & PCI
CORONARY_ANGIOGRAPHY_CONCEPTS = {
    4142645,  # Angiography of coronary artery
    4223020,  # Cardiac catheterization
    4216130,  # Percutaneous coronary intervention
    4301351,  # Coronary angiography (procedure)
    4180638,  # Cardiac catheterization with coronary angiography
    4052536   # Diagnostic coronary arteriography
}

# Comorbidities / Conditions
DIABETES_CONCEPTS = {201826, 201254, 443238, 4193704}
HEART_FAILURE_CONCEPTS = {316139} # Standard HF concept, placeholder if not in list
CKD_CONCEPTS = {46271022, 443597} # General CKD concepts

# Outcomes
AKI_CONCEPTS = {761083, 197320, 40481064, 4328366, 37116432, 45757442, 37016366}
DIALYSIS_CONCEPTS = {4032243, 4146536, 4324124, 4019967, 40482357}

# Analysis Settings
SEED = 42
N_FOLDS = 5
TRIM_QUANTILES = (0.025, 0.975)
OUTCOME_WINDOW_DAYS = 3  # 48-72h window (approx 3 days)

# =============================================================================
# 2. DATA LOADING & PREPARATION
# =============================================================================

def load_checkpoint_data():
    """Loads pre-computed checkpoint files from disk."""
    print("Loading checkpoint data...")
    try:
        df = pd.read_parquet("df_cohort.parquet")
        X = sp.load_npz("X_sparse.npz")
        pid_map = joblib.load("pid_to_idx.joblib")
        feat_map = joblib.load("feat_to_idx.joblib")
        
        print(f"Loaded df_cohort: {df.shape}")
        print(f"Loaded X_sparse: {X.shape}")
        return df, X, pid_map, feat_map
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Checkpoint files not found. {e}")
        sys.exit(1)

def align_sparse_matrix(df_subset, X_full, pid_map):
    """
    Aligns the sparse matrix to a subset of the cohort dataframe.
    """
    # Get row indices for the subset
    # pid_map maps person_id -> original row index
    subset_indices = [pid_map[pid] for pid in df_subset.index if pid in pid_map]
    
    if len(subset_indices) != len(df_subset):
        print(f"WARNING: {len(df_subset) - len(subset_indices)} patients in dataframe not found in sparse matrix mapping.")
        # Filter dataframe to match available indices
        valid_pids = [pid for pid in df_subset.index if pid in pid_map]
        df_subset = df_subset.loc[valid_pids]
        subset_indices = [pid_map[pid] for pid in valid_pids]

    X_subset = X_full[subset_indices, :]
    return df_subset, X_subset

def identify_treatment_from_sparse(X, feat_map, iodixanol_id, ioversol_id):
    """
    Identifies treatment status based on sparse drug features.
    Returns:
        T: Binary array (1=Iodixanol, 0=Ioversol, -1=Excluded)
    """
    # Find column indices for the drugs
    # Features are typically named "DRUG_{concept_id}"
    iodixanol_feats = [f for f in feat_map.keys() if str(iodixanol_id) in f]
    ioversol_feats = [f for f in feat_map.keys() if str(ioversol_id) in f]
    
    print(f"Found Iodixanol features: {iodixanol_feats}")
    print(f"Found Ioversol features: {ioversol_feats}")
    
    if not iodixanol_feats or not ioversol_feats:
        print("WARNING: Contrast agent features not found in sparse matrix. Cannot assign treatment.")
        return np.full(X.shape[0], -1)

    iodixanol_cols = [feat_map[f] for f in iodixanol_feats]
    ioversol_cols = [feat_map[f] for f in ioversol_feats]
    
    # Sum occurrences (in case of multiple formulations)
    has_iodixanol = np.array(X[:, iodixanol_cols].sum(axis=1)).flatten() > 0
    has_ioversol = np.array(X[:, ioversol_cols].sum(axis=1)).flatten() > 0
    
    # Define T
    # T=1: Iodixanol ONLY
    # T=0: Ioversol ONLY
    # Exclude: Both or Neither
    
    T = np.full(X.shape[0], -1)
    
    mask_treated = has_iodixanol & (~has_ioversol)
    mask_control = has_ioversol & (~has_iodixanol)
    
    T[mask_treated] = 1
    T[mask_control] = 0
    
    print(f"Treatment Assignment: Iodixanol (T=1): {mask_treated.sum()}, Ioversol (T=0): {mask_control.sum()}")
    print(f"Excluded (Mixed/Neither): {len(T) - mask_treated.sum() - mask_control.sum()}")
    
    return T

# =============================================================================
# 3. COHORT SELECTION (VALOR TRIAL CRITERIA)
# =============================================================================

def apply_eligibility_criteria(df):
    """
    Applies VALOR trial eligibility criteria:
    1. Age >= 18
    2. Procedure = Coronary Angiography
    3. eGFR 20-59 (CKD Stages 3-4)
    4. No prior dialysis (assumed handled by cohort construction or checked here)
    """
    print("\nApplying Eligibility Criteria...")
    initial_n = len(df)
    
    # 1. Age >= 18
    df = df[df['age'] >= 18]
    print(f"  After Age >= 18: {len(df)} (Dropped {initial_n - len(df)})")
    
    # 2. Procedure = Coronary Angiography
    # Check if procedure_concept_id is in our set
    df = df[df['procedure_concept_id'].isin(CORONARY_ANGIOGRAPHY_CONCEPTS)]
    print(f"  After Procedure Check: {len(df)}")
    
    # 3. eGFR 20-59
    # Using baseline_egfr column
    df = df[(df['baseline_egfr'] >= 20) & (df['baseline_egfr'] < 60)]
    print(f"  After eGFR 20-59: {len(df)}")
    
    # 4. Exclude prior dialysis
    # If date_NEW_DIALYSIS_90 is present, ensure it's not BEFORE index
    if 'date_NEW_DIALYSIS_90' in df.columns:
        # Assuming NaT means no dialysis, or date > index means incident
        mask_prior_dialysis = (df['date_NEW_DIALYSIS_90'] <= df['index_date'])
        df = df[~mask_prior_dialysis]
        print(f"  After Prior Dialysis Exclusion: {len(df)}")
        
    return df

# =============================================================================
# 4. SOTAstack AIPW IMPLEMENTATION
# =============================================================================

def get_super_learner():
    """
    Returns a simplified SuperLearner-style estimator.
    Using Random Forest as the primary engine for robustness in this script,
    as full stacking can be computationally expensive in a single script run.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED
    )

def get_propensity_model():
    """
    L1 Lasso Logistic Regression for Propensity Score (SOTAstack standard).
    """
    return LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        n_jobs=-1,
        random_state=SEED
    )

def run_cross_fitted_aipw(X, T, Y, n_folds=5):
    """
    Performs Cross-Fitted AIPW estimation.
    """
    print(f"\nRunning {n_folds}-Fold Cross-Fitted AIPW...")
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n = len(T)
    
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    pi_hat = np.zeros(n)
    eif_val = np.zeros(n)
    
    # For diagnostics
    ps_models = []
    
    for fold, (train_idx, eval_idx) in enumerate(kf.split(X, T)):
        print(f"  Processing Fold {fold+1}/{n_folds}...")
        
        X_train, X_eval = X[train_idx], X[eval_idx]
        T_train, T_eval = T[train_idx], T[eval_idx]
        Y_train, Y_eval = Y[train_idx], Y[eval_idx]
        
        # 1. Propensity Score Model
        ps_model = get_propensity_model()
        ps_model.fit(X_train, T_train)
        pi_chunk = ps_model.predict_proba(X_eval)[:, 1]
        
        # Trimming
        pi_chunk = np.clip(pi_chunk, TRIM_QUANTILES[0], TRIM_QUANTILES[1])
        pi_hat[eval_idx] = pi_chunk
        
        # 2. Outcome Models
        # Mu0: Train on Control (T=0)
        mu0_model = get_super_learner()
        mu0_model.fit(X_train[T_train == 0], Y_train[T_train == 0])
        mu0_chunk = mu0_model.predict_proba(X_eval)[:, 1]
        mu0_hat[eval_idx] = mu0_chunk
        
        # Mu1: Train on Treated (T=1)
        mu1_model = get_super_learner()
        mu1_model.fit(X_train[T_train == 1], Y_train[T_train == 1])
        mu1_chunk = mu1_model.predict_proba(X_eval)[:, 1]
        mu1_hat[eval_idx] = mu1_chunk
        
        # 3. EIF Calculation (Doubly Robust)
        # psi = (mu1 - mu0) + T(Y-mu1)/pi - (1-T)(Y-mu0)/(1-pi)
        term1 = mu1_chunk - mu0_chunk
        term2 = (T_eval * (Y_eval - mu1_chunk)) / pi_chunk
        term3 = ((1 - T_eval) * (Y_eval - mu0_chunk)) / (1 - pi_chunk)
        
        eif_val[eval_idx] = term1 + term2 - term3
        
    # Inference
    ate = np.mean(eif_val)
    se = np.std(eif_val) / np.sqrt(n)
    z_score = ate / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    # Risk Ratio (derived from marginal means)
    risk_1 = np.mean(mu1_hat)
    risk_0 = np.mean(mu0_hat)
    rr = risk_1 / risk_0 if risk_0 > 0 else np.nan
    
    results = {
        'ATE': ate,
        'SE': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'P_Value': p_value,
        'Risk_1': risk_1,
        'Risk_0': risk_0,
        'RR': rr,
        'PS': pi_hat
    }
    
    return results

# =============================================================================
# 5. DIAGNOSTICS
# =============================================================================

def calculate_diagnostics(X, T, ps):
    """Calculates SMD, ESS, and Overlap."""
    print("\nCalculating Diagnostics...")
    
    # 1. Effective Sample Size (ESS)
    p_t = T.mean()
    weights = np.where(T == 1, 1/ps, 1/(1-ps))
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    ess_ratio = ess / len(T)
    
    # 2. Overlap Coefficient
    # Simple histogram overlap
    hist1, _ = np.histogram(ps[T==1], bins=20, range=(0,1), density=True)
    hist0, _ = np.histogram(ps[T==0], bins=20, range=(0,1), density=True)
    overlap = np.sum(np.minimum(hist1, hist0)) * (1/20)
    
    # 3. Sparse SMD (Simplified for top features)
    # Calculate unweighted means
    mean_1 = np.array(X[T==1].mean(axis=0)).flatten()
    mean_0 = np.array(X[T==0].mean(axis=0)).flatten()
    var_1 = np.array(X[T==1].var(axis=0)).flatten()
    var_0 = np.array(X[T==0].var(axis=0)).flatten()
    
    pooled_sd = np.sqrt((var_1 + var_0) / 2)
    pooled_sd[pooled_sd == 0] = 1e-6
    
    smds = np.abs(mean_1 - mean_0) / pooled_sd
    max_smd = np.max(smds)
    mean_smd = np.mean(smds)
    
    print(f"  ESS: {ess:.1f} (Ratio: {ess_ratio:.3f})")
    print(f"  Overlap Coefficient: {overlap:.3f}")
    print(f"  Max Unadjusted SMD: {max_smd:.3f}")
    
    return {'ESS': ess, 'Overlap': overlap, 'Max_SMD': max_smd}

def calculate_e_value(rr):
    """Calculates E-Value for Risk Ratio."""
    if pd.isna(rr) or rr <= 0: return 1.0
    if rr <= 1:
        rr_star = 1 / rr
    else:
        rr_star = rr
    return rr_star + np.sqrt(rr_star * (rr_star - 1))

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def main():
    print("=== STARTING VALOR TRIAL EMULATION (SOTAstack) ===")
    
    # 1. Load Data
    df_cohort, X_sparse, pid_to_idx, feat_to_idx = load_checkpoint_data()
    
    # 2. Apply Eligibility Criteria (VALOR: CKD + Coronary Angio)
    df_eligible = apply_eligibility_criteria(df_cohort)
    
    if len(df_eligible) < 100:
        print("CRITICAL: Cohort too small after eligibility filtering.")
        return

    # 3. Align Sparse Matrix
    df_aligned, X_aligned = align_sparse_matrix(df_eligible, X_sparse, pid_to_idx)
    
    # 4. Define Treatment (Iodixanol vs Ioversol)
    # Using sparse features to identify specific drugs
    T_full = identify_treatment_from_sparse(
        X_aligned, feat_to_idx, IODIXANOL_CONCEPT_ID, IOVERSOL_CONCEPT_ID
    )
    
    # Filter to rows where Treatment is defined (0 or 1)
    mask_valid_t = T_full != -1
    df_final = df_aligned[mask_valid_t].copy()
    X_final = X_aligned[mask_valid_t]
    T_final = T_full[mask_valid_t]
    
    print(f"\nFinal Analysis Cohort N: {len(df_final)}")
    print(f"  Iodixanol (Active): {np.sum(T_final == 1)}")
    print(f"  Ioversol (Comparator): {np.sum(T_final == 0)}")
    
    if len(df_final) < 50:
        print("CRITICAL: Insufficient sample size after treatment assignment.")
        return

    # 5. Define Outcome (CIN within 48-72h)
    # Using date_AKI_30 as proxy for event, checking window
    # Y=1 if AKI date is within [index, index + 3 days]
    if 'date_AKI_30' in df_final.columns:
        days_to_aki = (df_final['date_AKI_30'] - df_final['index_date']).dt.days
        Y_final = ((days_to_aki >= 0) & (days_to_aki <= OUTCOME_WINDOW_DAYS)).astype(int).values
        print(f"  Outcome Events (CIN/AKI <= {OUTCOME_WINDOW_DAYS}d): {sum(Y_final)} ({sum(Y_final)/len(Y_final):.2%})")
    else:
        print("CRITICAL: Outcome column 'date_AKI_30' not found.")
        return

    # 6. Run AIPW
    results = run_cross_fitted_aipw(X_final, T_final, Y_final, n_folds=N_FOLDS)
    
    # 7. Diagnostics
    diag = calculate_diagnostics(X_final, T_final, results['PS'])
    e_value = calculate_e_value(results['RR'])
    
    # 8. Regulatory Conclusion
    print("\n" + "="*60)
    print("REGULATORY CONCLUSION REPORT")
    print("="*60)
    print(f"Trial: VALOR Emulation (Iodixanol vs Ioversol in CKD)")
    print(f"Population: N={len(df_final)} (Active={np.sum(T_final==1)}, Comp={np.sum(T_final==0)})")
    print(f"Outcome: CIN/AKI within {OUTCOME_WINDOW_DAYS} days")
    print("-" * 60)
    print(f"Risk Difference (ATE): {results['ATE']:.4f} (95% CI: {results['CI_Lower']:.4f}, {results['CI_Upper']:.4f})")
    print(f"Risk Ratio (RR):       {results['RR']:.4f}")
    print(f"P-Value:               {results['P_Value']:.4f}")
    print(f"E-Value:               {e_value:.2f}")
    print("-" * 60)
    
    # Interpretation Logic
    if results['P_Value'] < 0.05:
        if results['ATE'] < 0:
            conclusion = "EVIDENCE OF BENEFIT: Iodixanol is associated with significantly lower risk of CIN compared to Ioversol."
        else:
            conclusion = "EVIDENCE OF HARM: Iodixanol is associated with significantly higher risk of CIN compared to Ioversol."
    else:
        conclusion = "NO SIGNIFICANT DIFFERENCE: Evidence does not support a difference between Iodixanol and Ioversol."
        
    print(f"CONCLUSION: {conclusion}")
    
    # Diagnostic Warnings
    if diag['ESS'] / len(df_final) < 0.2:
        print("WARNING: Low Effective Sample Size. Propensity overlap may be poor.")
    if diag['Overlap'] < 0.1:
        print("WARNING: Poor Propensity Score Overlap. Results rely on extrapolation.")
        
    print("="*60)

if __name__ == "__main__":
    main()