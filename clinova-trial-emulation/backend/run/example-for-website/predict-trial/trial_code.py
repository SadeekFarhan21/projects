import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & OMOP CONCEPT IDS
# =============================================================================

# Validated OMOP Concept IDs (from prompt)
IODIXANOL_ID = 19003201
IOPAMIDOL_ID = 19081224

# Diabetes Concepts (Condition)
DIABETES_ALL = {201826, 201254, 443238, 4193704}

# CKD Stage 3 Definition (eGFR limits)
EGFR_MIN = 30
EGFR_MAX = 59

# Outcome Concepts (AKI)
AKI_CONCEPTS = {761083, 197320, 40481064, 4328366, 37116432, 45757442, 37016366}

# Analysis Settings
RANDOM_STATE = 42
N_FOLDS = 5
PS_TRIM_LOW = 0.025
PS_TRIM_HIGH = 0.975

# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_checkpoint_data():
    """
    Loads pre-computed checkpoint data from the working directory.
    """
    print("Loading checkpoint data...")
    
    # In a real scenario, these files must exist. 
    # For this script to be runnable in a mock environment, we assume they are present.
    try:
        df_cohort = pd.read_parquet("df_cohort.parquet")
        X_sparse = sp.load_npz("X_sparse.npz")
        pid_to_idx = joblib.load("pid_to_idx.joblib")
        feat_to_idx = joblib.load("feat_to_idx.joblib")
        
        print(f"Data loaded: Cohort N={len(df_cohort)}, Features={X_sparse.shape[1]}")
        return df_cohort, X_sparse, pid_to_idx, feat_to_idx
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Checkpoint files not found. {e}")
        print("Ensure df_cohort.parquet, X_sparse.npz, pid_to_idx.joblib, feat_to_idx.joblib exist.")
        raise

# =============================================================================
# 3. COHORT DEFINITION & SUBSETTING
# =============================================================================

def identify_feature_presence(X_sparse, feat_to_idx, concept_ids, prefix_filter=None):
    """
    Identifies rows in sparse matrix that have specific concept IDs.
    Returns a boolean array aligned with the sparse matrix rows.
    """
    relevant_cols = []
    
    # Find column indices for the requested concepts
    for feat_name, idx in feat_to_idx.items():
        # Check if feature matches any concept ID
        # Feature names usually formatted like "COND_201826" or "DRUG_19003201"
        for cid in concept_ids:
            if str(cid) in feat_name:
                if prefix_filter and not feat_name.startswith(prefix_filter):
                    continue
                relevant_cols.append(idx)
    
    if not relevant_cols:
        return np.zeros(X_sparse.shape[0], dtype=bool)
    
    # Sum across relevant columns to find presence
    # Slicing sparse matrix: X[:, cols]
    presence_sum = X_sparse[:, relevant_cols].sum(axis=1)
    # Convert matrix to flat array
    presence_flat = np.array(presence_sum).flatten()
    
    return presence_flat > 0

def define_predict_trial_cohort(df_cohort, X_sparse, pid_to_idx, feat_to_idx):
    """
    Applies PREDICT trial eligibility criteria:
    1. Diabetes History
    2. CKD Stage 3 (eGFR 30-59)
    3. Exposure: Iodixanol (Intervention) vs Iopamidol (Comparator)
    """
    print("\nApplying PREDICT Trial Eligibility Criteria...")
    
    # 0. Track original row indices to slice X_sparse later
    # We assume df_cohort is aligned with X_sparse initially (0 to N-1)
    # If df_cohort index is person_id, we rely on implicit row ordering matching X_sparse
    # or use pid_to_idx to map.
    # Here we assume df_cohort rows 0..N correspond to X_sparse rows 0..N
    df_cohort['orig_row_idx'] = np.arange(len(df_cohort))
    
    # 1. Baseline Kidney Function: eGFR 30-59
    # Using dense column from df_cohort
    mask_ckd = (df_cohort['baseline_egfr'] >= EGFR_MIN) & (df_cohort['baseline_egfr'] <= EGFR_MAX)
    print(f"  - CKD Stage 3 (eGFR 30-59): {mask_ckd.sum()} patients")
    
    # 2. Diabetes Mellitus (History)
    # Using sparse features
    has_diabetes = identify_feature_presence(X_sparse, feat_to_idx, DIABETES_ALL, prefix_filter="COND")
    # Map boolean array back to df_cohort (assuming alignment)
    df_cohort['has_diabetes'] = has_diabetes
    mask_diabetes = df_cohort['has_diabetes'] == True
    print(f"  - Diabetes History: {mask_diabetes.sum()} patients")
    
    # 3. Treatment Identification (Iodixanol vs Iopamidol)
    # Note: We look for these drugs in the sparse matrix (Drug Exposure)
    # Ideally, we'd check if they occurred ON the index date, but for this emulation
    # we assume the 'index_date' in df_cohort corresponds to the contrast procedure
    # and we look for the drug record.
    
    has_iodixanol = identify_feature_presence(X_sparse, feat_to_idx, {IODIXANOL_ID}, prefix_filter="DRUG")
    has_iopamidol = identify_feature_presence(X_sparse, feat_to_idx, {IOPAMIDOL_ID}, prefix_filter="DRUG")
    
    df_cohort['exp_iodixanol'] = has_iodixanol
    df_cohort['exp_iopamidol'] = has_iopamidol
    
    # Mutually exclusive groups for clean comparison (exclude mixed exposure)
    mask_iodixanol = (df_cohort['exp_iodixanol']) & (~df_cohort['exp_iopamidol'])
    mask_iopamidol = (df_cohort['exp_iopamidol']) & (~df_cohort['exp_iodixanol'])
    
    print(f"  - Iodixanol (Intervention): {mask_iodixanol.sum()} patients")
    print(f"  - Iopamidol (Comparator): {mask_iopamidol.sum()} patients")
    
    # Combine Filters
    mask_eligible = mask_ckd & mask_diabetes & (mask_iodixanol | mask_iopamidol)
    
    df_final = df_cohort[mask_eligible].copy()
    
    # Define Treatment Variable T (1=Iodixanol, 0=Iopamidol)
    df_final['T'] = df_final['exp_iodixanol'].astype(int)
    
    print(f"Final Study Cohort N: {len(df_final)}")
    print(f"  - Treated (Iodixanol): {df_final['T'].sum()}")
    print(f"  - Control (Iopamidol): {len(df_final) - df_final['T'].sum()}")
    
    # Slice Sparse Matrix to match filtered cohort
    # Use the tracked original row indices
    subset_indices = df_final['orig_row_idx'].values
    X_final = X_sparse[subset_indices]
    
    return df_final, X_final

# =============================================================================
# 4. OUTCOME DEFINITION
# =============================================================================

def define_outcome(df, index_col='index_date', outcome_date_col='date_AKI_30'):
    """
    Defines the primary outcome: AKI within 48-72 hours.
    
    Logic:
    - Calculate days between index_date and outcome_date
    - If days in [2, 3], Y = 1
    - Else Y = 0
    """
    # Ensure datetime
    df[index_col] = pd.to_datetime(df[index_col])
    df[outcome_date_col] = pd.to_datetime(df[outcome_date_col])
    
    # Calculate time to event
    time_to_event = (df[outcome_date_col] - df[index_col]).dt.days
    
    # Primary Window: 48h to 72h (Days 2 and 3)
    # Note: Day 0 is index. Day 1 is 24h. Day 2 is 48h.
    mask_window = (time_to_event >= 2) & (time_to_event <= 3)
    
    Y = mask_window.astype(int).values
    
    print(f"\nOutcome Definition (AKI within 48-72h):")
    print(f"  - Events: {Y.sum()} ({Y.mean()*100:.2f}%)")
    
    return Y

# =============================================================================
# 5. SOTAstack ANALYSIS (PS + AIPW)
# =============================================================================

def fit_lsps_model(X, T):
    """
    Fits Large-Scale Propensity Score (LSPS) using L1-regularized Logistic Regression.
    """
    print("Fitting LSPS Model (L1 Lasso)...")
    # Using SAGA solver which handles sparse data and L1 penalty efficiently
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.1,  # Moderate regularization
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, T)
    return model

def calculate_smd(X, T, weights):
    """
    Calculates Standardized Mean Difference (SMD) for sparse data.
    """
    # Weighted means
    W = sp.diags(weights)
    X_w = W @ X
    
    # Split by group
    idx_1 = np.where(T == 1)[0]
    idx_0 = np.where(T == 0)[0]
    
    w_1 = weights[idx_1]
    w_0 = weights[idx_0]
    
    sum_w1 = np.sum(w_1)
    sum_w0 = np.sum(w_0)
    
    # Means
    mu_1 = np.array(X_w[idx_1].sum(axis=0) / sum_w1).flatten()
    mu_0 = np.array(X_w[idx_0].sum(axis=0) / sum_w0).flatten()
    
    # Variances (simplified pooled variance calculation for speed)
    # Var = E[x^2] - E[x]^2
    X2 = X.power(2)
    X2_w = W @ X2
    
    E_x2_1 = np.array(X2_w[idx_1].sum(axis=0) / sum_w1).flatten()
    E_x2_0 = np.array(X2_w[idx_0].sum(axis=0) / sum_w0).flatten()
    
    var_1 = E_x2_1 - mu_1**2
    var_0 = E_x2_0 - mu_0**2
    
    pooled_sd = np.sqrt((var_1 + var_0) / 2)
    pooled_sd[pooled_sd < 1e-6] = 1e-6 # Avoid div/0
    
    smd = np.abs(mu_1 - mu_0) / pooled_sd
    return smd

def run_cross_fitted_aipw(X, T, Y):
    """
    Implements Cross-Fitted AIPW (Augmented Inverse Probability Weighting).
    Uses L1 Lasso for both PS and Outcome models (SOTAstack compliant).
    """
    print(f"\nRunning {N_FOLDS}-Fold Cross-Fitted AIPW...")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Storage for influence functions
    eif_values = np.zeros(len(T))
    mu1_preds = np.zeros(len(T))
    mu0_preds = np.zeros(len(T))
    ps_preds = np.zeros(len(T))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, T)):
        print(f"  - Processing Fold {fold+1}/{N_FOLDS}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        T_train, T_val = T[train_idx], T[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        # 1. Propensity Score Model (P(T=1|X))
        ps_model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=500, class_weight='balanced')
        ps_model.fit(X_train, T_train)
        pi_hat = ps_model.predict_proba(X_val)[:, 1]
        
        # Trimming
        pi_hat = np.clip(pi_hat, PS_TRIM_LOW, PS_TRIM_HIGH)
        ps_preds[val_idx] = pi_hat
        
        # 2. Outcome Models (E[Y|X, T=0] and E[Y|X, T=1])
        # Model for T=0
        mask_0 = T_train == 0
        mu0_model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=500)
        mu0_model.fit(X_train[mask_0], Y_train[mask_0])
        mu0_hat = mu0_model.predict_proba(X_val)[:, 1]
        mu0_preds[val_idx] = mu0_hat
        
        # Model for T=1
        mask_1 = T_train == 1
        mu1_model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=500)
        mu1_model.fit(X_train[mask_1], Y_train[mask_1])
        mu1_hat = mu1_model.predict_proba(X_val)[:, 1]
        mu1_preds[val_idx] = mu1_hat
        
        # 3. Calculate Efficient Influence Function (EIF)
        # EIF = (mu1 - mu0) + T(Y - mu1)/pi - (1-T)(Y - mu0)/(1-pi)
        
        term1 = mu1_hat - mu0_hat
        term2 = (T_val * (Y_val - mu1_hat)) / pi_hat
        term3 = ((1 - T_val) * (Y_val - mu0_hat)) / (1 - pi_hat)
        
        eif_values[val_idx] = term1 + term2 - term3

    # Aggregate Results
    ate = np.mean(eif_values)
    se = np.std(eif_values) / np.sqrt(len(T))
    
    # Risk Ratio Calculation (using mean potential outcomes)
    risk1 = np.mean(mu1_preds)
    risk0 = np.mean(mu0_preds)
    rr = risk1 / risk0 if risk0 > 0 else np.nan
    
    # Confidence Intervals
    z_score = 1.96
    ci_lower = ate - z_score * se
    ci_upper = ate + z_score * se
    p_value = 2 * (1 - norm.cdf(np.abs(ate / se)))
    
    results = {
        'ATE': ate,
        'SE': se,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'P_Value': p_value,
        'Risk_1': risk1,
        'Risk_0': risk0,
        'RR': rr
    }
    
    return results, ps_preds

# =============================================================================
# 6. DIAGNOSTICS & REPORTING
# =============================================================================

def calculate_e_value(rr):
    """Calculates E-Value for Risk Ratio."""
    if rr <= 1:
        return 1
    return rr + np.sqrt(rr * (rr - 1))

def generate_conclusion(results):
    """Generates regulatory conclusion based on results."""
    ate = results['ATE']
    p_val = results['P_Value']
    ci_low = results['CI_Lower']
    ci_high = results['CI_Upper']
    
    print("\n" + "="*60)
    print("REGULATORY CONCLUSION")
    print("="*60)
    print(f"Comparison: Iodixanol (Iso-osmolar) vs Iopamidol (Low-osmolar)")
    print(f"Population: Diabetic CKD Stage 3")
    print(f"Outcome:    AKI within 48-72h")
    print("-" * 60)
    print(f"ATE (Risk Diff): {ate:.4f} (95% CI: {ci_low:.4f} to {ci_high:.4f})")
    print(f"Risk Ratio:      {results['RR']:.4f}")
    print(f"P-Value:         {p_val:.4f}")
    print("-" * 60)
    
    if p_val < 0.05:
        if ate < 0:
            print("CONCLUSION: EVIDENCE OF BENEFIT.")
            print("Iodixanol is associated with a statistically significant REDUCTION")
            print("in AKI risk compared to Iopamidol in this population.")
        else:
            print("CONCLUSION: EVIDENCE OF HARM.")
            print("Iodixanol is associated with a statistically significant INCREASE")
            print("in AKI risk compared to Iopamidol.")
    else:
        print("CONCLUSION: NO SIGNIFICANT DIFFERENCE.")
        print("The analysis does not reject the null hypothesis.")
        print("There is insufficient evidence to claim a safety advantage for Iodixanol.")
        
        # Non-inferiority check (hypothetical margin of 0.02)
        if ci_high < 0.02:
            print("However, results may support non-inferiority (upper CI < 2%).")

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    print("Starting PREDICT Trial Emulation (SOTAstack Implementation)...")
    
    # 1. Load Data
    try:
        df_cohort, X_sparse, pid_to_idx, feat_to_idx = load_checkpoint_data()
    except Exception as e:
        print("Aborting: Data load failed.")
        return

    # 2. Define Cohort (Eligibility)
    df_final, X_final = define_predict_trial_cohort(df_cohort, X_sparse, pid_to_idx, feat_to_idx)
    
    if len(df_final) < 100:
        print("CRITICAL: Cohort too small for analysis (<100 patients).")
        return

    # 3. Define Variables
    T = df_final['T'].values
    Y = define_outcome(df_final)
    
    # 4. Run Analysis (AIPW)
    results, ps_scores = run_cross_fitted_aipw(X_final, T, Y)
    
    # 5. Diagnostics
    # Calculate IPTW weights for diagnostics
    p_t = T.mean()
    weights = np.where(T == 1, 1/ps_scores, 1/(1-ps_scores))
    
    # ESS
    ess = (np.sum(weights)**2) / np.sum(weights**2)
    print(f"\nDiagnostics:")
    print(f"  - Effective Sample Size (ESS): {ess:.1f} (Ratio: {ess/len(T):.2f})")
    
    # SMD (Sparse)
    # Note: Calculating SMD on all 70k features is heavy. 
    # In production, we'd select top features. Here we skip full SMD print for brevity
    # but the function is implemented above.
    
    # E-Value
    rr_eval = results['RR'] if results['RR'] > 1 else (1/results['RR'] if results['RR'] > 0 else 1)
    e_val = calculate_e_value(rr_eval)
    print(f"  - E-Value (Sensitivity to Unmeasured Confounding): {e_val:.2f}")

    # 6. Conclusion
    generate_conclusion(results)

if __name__ == "__main__":
    main()