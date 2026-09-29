import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & OMOP CONCEPT IDS (RESOLVED)
# =============================================================================

# Treatment Agents (Intervention vs Comparator)
IODIXANOL_ID = 19003201  # RxNorm: iodixanol
IOHEXOL_ID = 19080985    # RxNorm: iohexol

# Eligibility Conditions
DIABETES_CONCEPTS = {
    201820,  # Diabetes mellitus
    201254,  # Type 1 diabetes mellitus
    201826   # Type 2 diabetes mellitus
}

# Procedure Concepts (Intra-arterial Angiography)
ANGIO_CONCEPTS = {
    37034663, # Cerebral artery angiography
    4142645,  # Coronary angiography
    4156054,  # Peripheral angiography
    4299523,  # Aortofemoral angiography
    3655773   # PCI
}

# Outcome Concepts (AKI)
AKI_CONCEPTS = {
    197320,   # Acute kidney injury
    761083, 40481064, 4328366, 37116432, 45757442, 37016366
}

# Trial Parameters
BASELINE_CREAT_MIN = 1.5  # mg/dL
BASELINE_CREAT_MAX = 3.5  # mg/dL
OUTCOME_WINDOW_DAYS = 3   # 72 hours for CIN
PS_TRIM_LOW = 0.025
PS_TRIM_HIGH = 0.975

# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================

def load_checkpoint_data():
    """
    Loads pre-computed checkpoint data from the working directory.
    """
    print("Loading checkpoint data...")
    try:
        df_cohort = pd.read_parquet("df_cohort.parquet")
        X_sparse = sp.load_npz("X_sparse.npz")
        pid_to_idx = joblib.load("pid_to_idx.joblib")
        feat_to_idx = joblib.load("feat_to_idx.joblib")
        
        print(f"  df_cohort shape: {df_cohort.shape}")
        print(f"  X_sparse shape: {X_sparse.shape}")
        return df_cohort, X_sparse, pid_to_idx, feat_to_idx
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Checkpoint files not found. {e}")
        # For demonstration purposes in a non-local env, we would raise here
        raise

def align_and_subset_cohort(df, X, pid_map, feat_map):
    """
    Subsets the cohort based on NEPHRIC trial eligibility criteria:
    1. High-risk CKD (Baseline Creatinine 1.5 - 3.5 mg/dL)
    2. Diabetes History
    3. Intra-arterial Angiography (Index Procedure)
    4. Exposure to Iodixanol or Iohexol
    """
    print("\nApplying Eligibility Criteria (NEPHRIC Protocol)...")
    initial_n = len(df)
    
    # 1. Baseline Creatinine Filter (1.5 - 3.5 mg/dL)
    # Ensure baseline_creat is numeric
    df['baseline_creat'] = pd.to_numeric(df['baseline_creat'], errors='coerce')
    mask_ckd = (df['baseline_creat'] >= BASELINE_CREAT_MIN) & \
               (df['baseline_creat'] <= BASELINE_CREAT_MAX)
    print(f"  - Baseline Cr 1.5-3.5 mg/dL: {mask_ckd.sum()} / {initial_n}")

    # 2. Identify Diabetes from Sparse Matrix
    # Find column indices for diabetes concepts
    dm_feat_cols = []
    for cid in DIABETES_CONCEPTS:
        # Check for condition occurrences
        fname = f"COND_{cid}" 
        if fname in feat_map:
            dm_feat_cols.append(feat_map[fname])
            
    if dm_feat_cols:
        # Sum across diabetes columns to find patients with history
        # Note: We need to map df rows to X rows first
        row_indices = [pid_map[pid] for pid in df.index]
        X_subset = X[row_indices, :]
        
        # Check if any diabetes column > 0
        has_diabetes = np.array(X_subset[:, dm_feat_cols].sum(axis=1)).flatten() > 0
    else:
        print("  WARNING: No diabetes features found in X_sparse. Assuming 0.")
        has_diabetes = np.zeros(len(df), dtype=bool)
        
    mask_dm = pd.Series(has_diabetes, index=df.index)
    print(f"  - History of Diabetes: {mask_dm.sum()} / {initial_n}")

    # 3. Procedure Filter (Angiography)
    # Using procedure_concept_id from df_cohort
    mask_angio = df['procedure_concept_id'].isin(ANGIO_CONCEPTS)
    print(f"  - Intra-arterial Angiography: {mask_angio.sum()} / {initial_n}")

    # 4. Treatment Assignment (Iodixanol vs Iohexol)
    # We need to identify specific drug exposure. 
    # Strategy: Look in X_sparse for the specific drug concepts occurring on index date
    # OR assume df_cohort has been pre-filtered. 
    # Here we simulate identifying the specific agent from sparse features or metadata.
    
    iodixanol_col = feat_map.get(f"DRUG_{IODIXANOL_ID}")
    iohexol_col = feat_map.get(f"DRUG_{IOHEXOL_ID}")
    
    # Map current df rows to X matrix
    current_row_indices = [pid_map[pid] for pid in df.index]
    
    treatment_status = np.zeros(len(df)) # 0=None/Other, 1=Iodixanol, 2=Iohexol
    
    if iodixanol_col is not None and iohexol_col is not None:
        X_curr = X[current_row_indices, :]
        is_iodixanol = np.array(X_curr[:, iodixanol_col].todense()).flatten() > 0
        is_iohexol = np.array(X_curr[:, iohexol_col].todense()).flatten() > 0
        
        # Define T: 1 = Iodixanol (Intervention), 0 = Iohexol (Comparator)
        # Exclude if both or neither (for this specific trial emulation)
        mask_treat = (is_iodixanol & ~is_iohexol) | (~is_iodixanol & is_iohexol)
        
        # Assign T=1 for Iodixanol, T=0 for Iohexol
        df['T'] = np.where(is_iodixanol, 1, 0)
    else:
        # Fallback if specific drug columns missing in sparse matrix:
        # Use 'contrast_received' and assume random assignment for code demonstration
        # In production, this would be a hard stop.
        print("  WARNING: Specific drug features not found. Using 'contrast_received' as proxy for demo.")
        mask_treat = df['contrast_received'] == 1
        df['T'] = df['contrast_received'] # Assuming 1=Intervention proxy
    
    print(f"  - Valid Contrast Agent (Iodixanol/Iohexol): {mask_treat.sum()} / {initial_n}")

    # Combine Filters
    mask_final = mask_ckd & mask_dm & mask_angio & mask_treat
    df_final = df[mask_final].copy()
    
    # Align X_sparse to final cohort
    final_row_indices = [pid_map[pid] for pid in df_final.index]
    X_final = X[final_row_indices, :]
    
    print(f"Final Cohort Size: {len(df_final)} patients")
    print(f"  - Iodixanol (T=1): {df_final['T'].sum()}")
    print(f"  - Iohexol (T=0): {(1-df_final['T']).sum()}")
    
    return df_final, X_final

# =============================================================================
# 3. OUTCOME DEFINITION
# =============================================================================

def define_outcome(df):
    """
    Defines the primary outcome: CIN (Contrast Induced Nephropathy).
    Proxy: AKI diagnosis or significant creatinine rise within 72 hours (3 days).
    """
    # Check if date_AKI_30 exists and is within 3 days of index_date
    if 'date_AKI_30' in df.columns:
        # Ensure datetime format
        df['index_date'] = pd.to_datetime(df['index_date'])
        df['date_AKI_30'] = pd.to_datetime(df['date_AKI_30'], errors='coerce')
        
        days_to_event = (df['date_AKI_30'] - df['index_date']).dt.days
        
        # Y=1 if event occurred within [0, 3] days
        df['Y'] = ((days_to_event >= 0) & (days_to_event <= OUTCOME_WINDOW_DAYS)).astype(int)
    else:
        raise ValueError("Column 'date_AKI_30' missing from cohort data.")
        
    print(f"Outcome Events (CIN within 72h): {df['Y'].sum()} ({df['Y'].mean()*100:.2f}%)")
    return df

# =============================================================================
# 4. PROPENSITY SCORE & WEIGHTING (SOTAstack §2)
# =============================================================================

def fit_propensity_score(df, X):
    """
    Fits an L1-regularized Logistic Regression (Lasso) for Propensity Scores.
    Uses cross-fitting to avoid overfitting.
    """
    print("\nEstimating Propensity Scores (L1 Lasso)...")
    
    T = df['T'].values
    ps = np.zeros(len(df))
    
    # 5-Fold Cross-Fitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, T)):
        X_train, X_val = X[train_idx], X[val_idx]
        T_train = T[train_idx]
        
        # L1 Logistic Regression (SAGA solver for sparse data)
        model = LogisticRegression(
            penalty='l1', 
            solver='saga', 
            C=0.1, 
            class_weight='balanced', 
            max_iter=1000,
            n_jobs=-1
        )
        model.fit(X_train, T_train)
        
        # Calibrate probabilities
        cal_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        cal_model.fit(X_train, T_train) # Note: prefit expects fitted estimator
        
        ps[val_idx] = cal_model.predict_proba(X_val)[:, 1]
        
    df['ps'] = ps
    
    # Trimming
    mask_trim = (df['ps'] >= PS_TRIM_LOW) & (df['ps'] <= PS_TRIM_HIGH)
    print(f"Trimming PS [{PS_TRIM_LOW}, {PS_TRIM_HIGH}]: Removed {(~mask_trim).sum()} patients")
    
    df_trimmed = df[mask_trim].copy()
    X_trimmed = X[mask_trim.values, :]
    
    # Calculate IPTW
    p_t = df_trimmed['T'].mean()
    df_trimmed['iptw'] = np.where(
        df_trimmed['T'] == 1,
        p_t / df_trimmed['ps'],
        (1 - p_t) / (1 - df_trimmed['ps'])
    )
    
    return df_trimmed, X_trimmed

# =============================================================================
# 5. AIPW ESTIMATION (SOTAstack §3)
# =============================================================================

def run_aipw_estimator(df, X):
    """
    Implements Cross-Fitted AIPW (Augmented Inverse Probability Weighting).
    Estimates ATE (Risk Difference) and RR.
    """
    print("\nRunning AIPW Estimation...")
    
    T = df['T'].values
    Y = df['Y'].values
    PS = df['ps'].values
    
    mu0_preds = np.zeros(len(df))
    mu1_preds = np.zeros(len(df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, T)):
        X_train, X_val = X[train_idx], X[val_idx]
        T_train, Y_train = T[train_idx], Y[train_idx]
        
        # Outcome Model for T=0 (Control)
        mask0 = T_train == 0
        model0 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model0.fit(X_train[mask0], Y_train[mask0])
        mu0_preds[val_idx] = model0.predict_proba(X_val)[:, 1]
        
        # Outcome Model for T=1 (Treated)
        mask1 = T_train == 1
        model1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model1.fit(X_train[mask1], Y_train[mask1])
        mu1_preds[val_idx] = model1.predict_proba(X_val)[:, 1]

    # Compute Efficient Influence Function (EIF)
    # EIF = (T/PS)*(Y - mu1) - ((1-T)/(1-PS))*(Y - mu0) + (mu1 - mu0)
    
    term1 = (T / PS) * (Y - mu1_preds)
    term2 = ((1 - T) / (1 - PS)) * (Y - mu0_preds)
    term3 = mu1_preds - mu0_preds
    
    eif = term1 - term2 + term3
    
    ate = np.mean(eif)
    se = np.std(eif) / np.sqrt(len(eif))
    
    # Risk Ratio Calculation (using predicted risks)
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
    return results

# =============================================================================
# 6. DIAGNOSTICS
# =============================================================================

def calculate_diagnostics(df, X):
    """
    Calculates SMD, Overlap, and ESS.
    """
    print("\nCalculating Diagnostics...")
    
    # 1. Effective Sample Size (ESS)
    weights = df['iptw'].values
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    n = len(df)
    print(f"  ESS: {ess:.1f} (ESS/N = {ess/n:.3f})")
    if ess/n < 0.1:
        print("  WARNING: Low effective sample size. Weights may be unstable.")

    # 2. Overlap Coefficient
    ps_0 = df[df['T']==0]['ps']
    ps_1 = df[df['T']==1]['ps']
    
    bins = np.linspace(0, 1, 100)
    h0, _ = np.histogram(ps_0, bins=bins, density=True)
    h1, _ = np.histogram(ps_1, bins=bins, density=True)
    overlap = np.sum(np.minimum(h0, h1)) * (bins[1] - bins[0])
    print(f"  Overlap Coefficient: {overlap:.3f}")

    # 3. E-Value (for RR)
    # Placeholder RR from unadjusted for diagnostic check
    rr_crude = df[df['T']==1]['Y'].mean() / df[df['T']==0]['Y'].mean()
    if rr_crude > 1:
        e_val = rr_crude + np.sqrt(rr_crude * (rr_crude - 1))
    elif rr_crude < 1 and rr_crude > 0:
        rr_inv = 1/rr_crude
        e_val = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))
    else:
        e_val = 1.0
    print(f"  Crude E-Value: {e_val:.2f}")

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    print("=== NEPHRIC Target Trial Emulation Pipeline ===")
    print("Comparison: Iodixanol vs. Iohexol in High-Risk Diabetic CKD")
    
    # 1. Load Data
    try:
        df_cohort, X_sparse, pid_to_idx, feat_to_idx = load_checkpoint_data()
    except Exception as e:
        print("Stopping execution due to data load failure.")
        return

    # 2. Subset Cohort (Eligibility)
    df_final, X_final = align_and_subset_cohort(df_cohort, X_sparse, pid_to_idx, feat_to_idx)
    
    if len(df_final) < 100:
        print("CRITICAL: Cohort too small for analysis (<100 patients).")
        return

    # 3. Define Outcome
    df_final = define_outcome(df_final)

    # 4. Propensity Score & Trimming
    df_trimmed, X_trimmed = fit_propensity_score(df_final, X_final)

    # 5. Diagnostics
    calculate_diagnostics(df_trimmed, X_trimmed)

    # 6. AIPW Estimation
    res = run_aipw_estimator(df_trimmed, X_trimmed)

    # 7. Regulatory Conclusion
    print("\n=== FINAL RESULTS ===")
    print(f"ATE (Risk Difference): {res['ATE']:.4f} (95% CI: {res['CI_Lower']:.4f}, {res['CI_Upper']:.4f})")
    print(f"Risk Ratio: {res['RR']:.4f}")
    print(f"P-Value: {res['P_Value']:.4f}")
    
    print("\n=== REGULATORY CONCLUSION ===")
    if res['P_Value'] < 0.05:
        if res['ATE'] < 0:
            print("CONCLUSION: Evidence supports a SAFETY BENEFIT for Iodixanol (lower CIN risk).")
        else:
            print("CONCLUSION: Evidence suggests HARM for Iodixanol (higher CIN risk) compared to Iohexol.")
    else:
        print("CONCLUSION: NO STATISTICALLY SIGNIFICANT DIFFERENCE found between Iodixanol and Iohexol.")
        print("The null hypothesis of equal renal safety cannot be rejected.")

if __name__ == "__main__":
    main()