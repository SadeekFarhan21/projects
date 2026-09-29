import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import warnings
from datetime import timedelta
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & CONCEPT IDS
# =============================================================================

# Resolved OMOP Concept IDs (from Prompt)
IODIXANOL_ID = 19003201
IOPAMIDOL_ID = 19081224
AKI_CONDITION_ID = 197320
CKD_CONDITION_ID = 46271022
ESRD_CONDITION_ID = 193782
DIALYSIS_PROC_IDS = {4195714, 4247794, 36685851}

# Trial Parameters
TRIAL_NAME = "BARRETT_2006_EMULATION"
OUTCOME_WINDOW_DAYS = 3  # 72 hours
CKD_EGFR_THRESHOLD = 60.0
MIN_EVENTS_THRESHOLD = 10

# SOTAstack Model Hyperparameters
L1_LASSO_PARAMS = {
    'penalty': 'l1',
    'solver': 'saga',
    'C': 0.2,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'n_jobs': -1,
    'random_state': 42
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': 42
}

# =============================================================================
# 2. DATA LOADING & PRE-PROCESSING
# =============================================================================

def load_checkpoint_data():
    """
    Loads pre-computed checkpoint data from the working directory.
    """
    print(f"[{TRIAL_NAME}] Loading checkpoint data...")
    
    try:
        df_cohort = pd.read_parquet("df_cohort.parquet")
        X_sparse = sp.load_npz("X_sparse.npz")
        pid_to_idx = joblib.load("pid_to_idx.joblib")
        feat_to_idx = joblib.load("feat_to_idx.joblib")
        
        print(f"  - Cohort loaded: {df_cohort.shape[0]} patients")
        print(f"  - Sparse matrix loaded: {X_sparse.shape}")
        return df_cohort, X_sparse, pid_to_idx, feat_to_idx
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Checkpoint files not found. {e}")
        raise

def apply_eligibility_criteria(df):
    """
    Subsets the cohort based on the Barrett 2006 Trial criteria:
    1. CKD (eGFR < 60)
    2. Exposure to Iodixanol (Intervention) or Iopamidol (Comparator)
    """
    print(f"[{TRIAL_NAME}] Applying eligibility criteria...")
    initial_n = len(df)
    
    # 1. CKD Criterion (eGFR < 60)
    # Handling missing eGFR: If missing, we assume not eligible for this specific CKD trial 
    # unless we impute. For strict emulation, we require baseline eGFR.
    df_eligible = df[df['baseline_egfr'] < CKD_EGFR_THRESHOLD].copy()
    n_ckd = len(df_eligible)
    print(f"  - Met CKD criterion (eGFR < {CKD_EGFR_THRESHOLD}): {n_ckd} ({n_ckd/initial_n:.1%})")

    # 2. Treatment Identification
    # We need to define the Treatment variable (T) based on specific drugs.
    # Assuming 'procedure_concept_id' or a 'drug_concept_id' column contains the agent.
    # If specific columns aren't in the checkpoint, we use 'contrast_received' 
    # and assume the cohort was already filtered to these two agents, 
    # but here we implement the logic as if the column exists or we map it.
    
    # Logic: Create a 'treatment_arm' column. 
    # 1 = Iodixanol, 0 = Iopamidol. Exclude others.
    
    # Note: In a real run, we check if 'drug_concept_id' exists. 
    # If not, we rely on 'procedure_concept_id' if it maps to the drugs, 
    # or 'contrast_received' if the checkpoint is pre-scoped.
    # Here, we simulate the identification for robustness.
    
    if 'drug_concept_id' in df.columns:
        mask_iodixanol = df_eligible['drug_concept_id'] == IODIXANOL_ID
        mask_iopamidol = df_eligible['drug_concept_id'] == IOPAMIDOL_ID
    elif 'procedure_concept_id' in df.columns:
        # Sometimes drug admins are recorded as procedures
        mask_iodixanol = df_eligible['procedure_concept_id'] == IODIXANOL_ID
        mask_iopamidol = df_eligible['procedure_concept_id'] == IOPAMIDOL_ID
    else:
        # Fallback: Assume the checkpoint 'contrast_received' binary 
        # specifically compares these two for this exercise context.
        print("  ! Specific drug IDs not found in columns. Using 'contrast_received' as proxy (1=Intervention, 0=Comparator).")
        mask_iodixanol = df_eligible['contrast_received'] == 1
        mask_iopamidol = df_eligible['contrast_received'] == 0

    df_eligible['T'] = np.nan
    df_eligible.loc[mask_iodixanol, 'T'] = 1
    df_eligible.loc[mask_iopamidol, 'T'] = 0
    
    # Drop rows that are neither (if any)
    df_final = df_eligible.dropna(subset=['T']).copy()
    df_final['T'] = df_final['T'].astype(int)
    
    print(f"  - Final Cohort N: {len(df_final)}")
    print(f"    - Iodixanol (T=1): {sum(df_final['T']==1)}")
    print(f"    - Iopamidol (T=0): {sum(df_final['T']==0)}")
    
    return df_final

def align_sparse_matrix(df_subset, X_sparse, pid_to_idx):
    """
    Slices the large sparse matrix to match the subsetted cohort.
    """
    print(f"[{TRIAL_NAME}] Aligning sparse feature matrix...")
    
    # Get row indices for the subset
    # pid_to_idx maps person_id -> original row index in X_sparse
    subset_row_indices = [pid_to_idx[pid] for pid in df_subset['person_id']]
    
    X_subset = X_sparse[subset_row_indices]
    print(f"  - Aligned Matrix Shape: {X_subset.shape}")
    
    return X_subset

def define_outcome(df):
    """
    Defines the primary outcome: CA-AKI within 72 hours.
    Uses 'date_AKI_30' (from checkpoint) and checks if it is within 3 days of index.
    """
    # Calculate days to event
    # Ensure datetime types
    df['index_date'] = pd.to_datetime(df['index_date'])
    df['date_AKI_30'] = pd.to_datetime(df['date_AKI_30'], errors='coerce')
    
    # Time to event
    time_to_aki = (df['date_AKI_30'] - df['index_date']).dt.days
    
    # Binary Outcome Y: Event occurred within window (0 to 3 days)
    # Note: We assume date_AKI_30 captures the diagnosis date. 
    # Ideally, we use creatinine values, but we use the checkpoint's derived date here.
    df['Y'] = ((time_to_aki >= 0) & (time_to_aki <= OUTCOME_WINDOW_DAYS)).astype(int)
    
    events = df['Y'].sum()
    print(f"[{TRIAL_NAME}] Outcome Definition (CA-AKI <= 72h):")
    print(f"  - Total Events: {events} ({events/len(df):.2%})")
    
    return df

# =============================================================================
# 3. MODELING (SOTAstack)
# =============================================================================

def get_super_learner():
    """
    Returns a StackingClassifier combining Lasso and Random Forest.
    (SOTAstack Section 3)
    """
    estimators = [
        ('lasso', LogisticRegression(**L1_LASSO_PARAMS)),
        ('rf', RandomForestClassifier(**RF_PARAMS))
    ]
    
    # Meta-learner: Logistic Regression
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(solver='lbfgs'),
        cv=3,
        n_jobs=1,
        passthrough=False
    )
    return clf

def fit_propensity_score(X, T):
    """
    Estimates Propensity Score using L1 Lasso (LSPS).
    """
    print(f"[{TRIAL_NAME}] Estimating Propensity Scores (L1 Lasso)...")
    
    # Using simple L1 Lasso for PS as per SOTAstack Section 2
    ps_model = LogisticRegression(**L1_LASSO_PARAMS)
    
    # Cross-fitting PS is handled in the AIPW loop, but we fit a global one for diagnostics
    ps_model.fit(X, T)
    ps_global = ps_model.predict_proba(X)[:, 1]
    
    return ps_global

def run_cross_fitted_aipw(X, T, Y, n_folds=5):
    """
    Implements Cross-Fitted AIPW (Doubly Robust) estimation.
    (SOTAstack Section 3)
    """
    print(f"[{TRIAL_NAME}] Running {n_folds}-Fold Cross-Fitted AIPW...")
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n = len(T)
    
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    pi_hat = np.zeros(n)
    eif_vec = np.zeros(n)
    
    fold = 1
    for train_idx, eval_idx in kf.split(X, T):
        print(f"  - Processing Fold {fold}/{n_folds}...")
        
        X_train, X_eval = X[train_idx], X[eval_idx]
        T_train, T_eval = T[train_idx], T[eval_idx]
        Y_train, Y_eval = Y[train_idx], Y[eval_idx]
        
        # 1. Fit Propensity Score (Pi)
        # Using L1 Lasso for speed and sparsity handling
        ps_model = LogisticRegression(**L1_LASSO_PARAMS)
        ps_model.fit(X_train, T_train)
        pi_eval = ps_model.predict_proba(X_eval)[:, 1]
        
        # Trim PS to avoid division by zero (SOTAstack Section 9)
        pi_eval = np.clip(pi_eval, 0.025, 0.975)
        pi_hat[eval_idx] = pi_eval
        
        # 2. Fit Outcome Models (Mu)
        # Using SuperLearner (or just RF/Lasso) separately for T=0 and T=1
        
        # Mu0: Train on T=0
        mask0 = (T_train == 0)
        mu0_model = get_super_learner()
        mu0_model.fit(X_train[mask0], Y_train[mask0])
        mu0_eval = mu0_model.predict_proba(X_eval)[:, 1]
        mu0_hat[eval_idx] = mu0_eval
        
        # Mu1: Train on T=1
        mask1 = (T_train == 1)
        mu1_model = get_super_learner()
        mu1_model.fit(X_train[mask1], Y_train[mask1])
        mu1_eval = mu1_model.predict_proba(X_eval)[:, 1]
        mu1_hat[eval_idx] = mu1_eval
        
        fold += 1

    # 3. Compute Efficient Influence Function (EIF)
    # EIF = (mu1 - mu0) + T(Y - mu1)/pi - (1-T)(Y - mu0)/(1-pi)
    term1 = mu1_hat - mu0_hat
    term2 = (T * (Y - mu1_hat)) / pi_hat
    term3 = ((1 - T) * (Y - mu0_hat)) / (1 - pi_hat)
    eif_vec = term1 + term2 - term3
    
    # Inference
    ate = np.mean(eif_vec)
    se = np.std(eif_vec) / np.sqrt(n)
    z_score = ate / se if se > 0 else 0
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    # Risk Ratio Calculation (using plug-in means of mu)
    risk_1 = np.mean(mu1_hat)
    risk_0 = np.mean(mu0_hat)
    rr = risk_1 / risk_0 if risk_0 > 0 else np.nan
    
    results = {
        'ATE': ate,
        'SE': se,
        'P_Value': p_value,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Risk_1': risk_1,
        'Risk_0': risk_0,
        'RR': rr
    }
    
    return results, pi_hat

# =============================================================================
# 4. DIAGNOSTICS
# =============================================================================

def calculate_smd(X, T, weights):
    """
    Calculates Standardized Mean Difference (SMD) for sparse data.
    (SOTAstack Section 4B)
    """
    # Helper for weighted mean/var on sparse matrix
    def weighted_stats(X_sub, w_sub):
        w_norm = w_sub / w_sub.sum()
        # Mean
        # Reshape w for broadcasting: (N, 1)
        w_diag = sp.diags(w_norm)
        means = np.array(w_diag.dot(X_sub).sum(axis=0)).flatten()
        
        # Var: E[x^2] - (E[x])^2
        X2 = X_sub.power(2)
        means2 = np.array(w_diag.dot(X2).sum(axis=0)).flatten()
        vars_ = means2 - means**2
        return means, vars_

    mu1, var1 = weighted_stats(X[T==1], weights[T==1])
    mu0, var0 = weighted_stats(X[T==0], weights[T==0])
    
    pooled_sd = np.sqrt((var1 + var0) / 2)
    pooled_sd[pooled_sd == 0] = 1e-6
    
    smd = np.abs((mu1 - mu0) / pooled_sd)
    return smd

def run_diagnostics(X, T, ps):
    """
    Runs balance and positivity checks.
    """
    print(f"[{TRIAL_NAME}] Running Diagnostics...")
    
    # 1. Calculate IPTW weights for diagnostics
    p_t = T.mean()
    weights = np.where(T==1, 1/ps, 1/(1-ps))
    
    # 2. Effective Sample Size (ESS)
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    n = len(T)
    print(f"  - Effective Sample Size (ESS): {ess:.1f} (Ratio: {ess/n:.2f})")
    if ess/n < 0.1:
        print("    WARNING: Low ESS. Weights may be unstable.")

    # 3. Covariate Balance (SMD)
    # Calculate unweighted SMD
    smd_unweighted = calculate_smd(X, T, np.ones(n))
    # Calculate weighted SMD
    smd_weighted = calculate_smd(X, T, weights)
    
    max_smd_unw = np.max(smd_unweighted)
    max_smd_w = np.max(smd_weighted)
    mean_smd_w = np.mean(smd_weighted)
    
    print(f"  - Max SMD (Unadjusted): {max_smd_unw:.3f}")
    print(f"  - Max SMD (Adjusted):   {max_smd_w:.3f}")
    print(f"  - Mean SMD (Adjusted):  {mean_smd_w:.3f}")
    
    if max_smd_w > 0.1:
        print("    WARNING: Max SMD > 0.1. Covariate balance not fully achieved.")
    else:
        print("    SUCCESS: Covariate balance achieved (Max SMD < 0.1).")

    # 4. Overlap
    ps_0 = ps[T==0]
    ps_1 = ps[T==1]
    # Simple overlap check: range intersection
    overlap_min = max(ps_0.min(), ps_1.min())
    overlap_max = min(ps_0.max(), ps_1.max())
    print(f"  - PS Overlap Range: [{overlap_min:.3f}, {overlap_max:.3f}]")

# =============================================================================
# 5. REGULATORY CONCLUSION
# =============================================================================

def generate_conclusion(results):
    """
    Generates a regulatory-style conclusion based on the results.
    """
    ate = results['ATE']
    p_val = results['P_Value']
    ci_low = results['CI_Lower']
    ci_high = results['CI_Upper']
    rr = results['RR']
    
    print("\n" + "="*60)
    print("REGULATORY CONCLUSION")
    print("="*60)
    print(f"Trial: {TRIAL_NAME}")
    print(f"Comparison: Iodixanol (Iso-osmolar) vs Iopamidol (Low-osmolar)")
    print(f"Population: CKD (eGFR < {CKD_EGFR_THRESHOLD})")
    print(f"Outcome: CA-AKI within 72 hours")
    print("-" * 60)
    print(f"Primary Estimand (ATE - Risk Difference): {ate:.4f} (95% CI: {ci_low:.4f} to {ci_high:.4f})")
    print(f"Secondary Estimand (Risk Ratio): {rr:.4f}")
    print(f"P-Value: {p_val:.4f}")
    print("-" * 60)
    
    if p_val < 0.05:
        if ate < 0:
            conclusion = (
                "CONCLUSION: Evidence supports a statistically significant REDUCTION in CA-AKI risk "
                "with Iodixanol compared to Iopamidol in this CKD population."
            )
        else:
            conclusion = (
                "CONCLUSION: Evidence suggests a statistically significant INCREASE in CA-AKI risk "
                "with Iodixanol compared to Iopamidol (favoring Comparator)."
            )
    else:
        conclusion = (
            "CONCLUSION: No statistically significant difference in CA-AKI risk was observed "
            "between Iodixanol and Iopamidol (Fail to reject null)."
        )
        
    print(conclusion)
    print("="*60 + "\n")

# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def main():
    print(f"Starting Target Trial Emulation: {TRIAL_NAME}")
    
    # 1. Load Data
    df_cohort, X_sparse, pid_to_idx, feat_to_idx = load_checkpoint_data()
    
    # 2. Apply Eligibility (CKD + Specific Drugs)
    df_final = apply_eligibility_criteria(df_cohort)
    
    if len(df_final) < 50:
        print("CRITICAL: Sample size too small after eligibility criteria. Aborting analysis.")
        return

    # 3. Align Sparse Matrix
    X_final = align_sparse_matrix(df_final, X_sparse, pid_to_idx)
    
    # 4. Define Outcome
    df_final = define_outcome(df_final)
    
    # Extract vectors for analysis
    T_vec = df_final['T'].values
    Y_vec = df_final['Y'].values
    
    if Y_vec.sum() < MIN_EVENTS_THRESHOLD:
        print(f"CRITICAL: Too few outcome events ({Y_vec.sum()}). Aborting analysis.")
        return

    # 5. Run AIPW Analysis
    results, ps_scores = run_cross_fitted_aipw(X_final, T_vec, Y_vec, n_folds=5)
    
    # 6. Run Diagnostics
    run_diagnostics(X_final, T_vec, ps_scores)
    
    # 7. Generate Conclusion
    generate_conclusion(results)
    
    # Save results
    joblib.dump(results, f"{TRIAL_NAME}_results.joblib")
    print("Analysis complete. Results saved.")

if __name__ == "__main__":
    main()