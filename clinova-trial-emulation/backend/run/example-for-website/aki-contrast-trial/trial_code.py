"""
AKI-Contrast Trial Emulation - Target Trial Analysis Code
Compares contrast-enhanced vs non-contrast CT/MRI imaging for AKI risk.
Using All of Us Research Program data with OMOP CDM.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION & OMOP CONCEPT IDS
# =============================================================================

# Contrast-Enhanced CT Procedures
CONTRAST_CT_CONCEPTS = {4139745, 21492176, 4335400, 3047782, 4327032,
                        3013610, 36713226, 3053128, 4252907, 3019625}

# Contrast-Enhanced MRI Procedures
CONTRAST_MRI_CONCEPTS = {4335399, 4161393, 4202274, 4197203, 36717294,
                         45765683, 37117806, 37109194, 37109196}

# Non-Contrast CT Procedures
NON_CONTRAST_CT_CONCEPTS = {37109313, 3049940, 37117305, 3047921, 36713200,
                            3018999, 40771605, 36713202, 3035568}

# Non-Contrast MRI Procedures
NON_CONTRAST_MRI_CONCEPTS = {37109312, 36713204, 36713045, 36713262, 3024397,
                             36713243, 3053040, 37109329, 42535581, 42535582}

# AKI Outcome Concepts
AKI_CONCEPTS = {761083, 197320, 40481064, 4328366, 37116432, 45757442, 37016366}

# Analysis Settings
RANDOM_STATE = 42
N_FOLDS = 5
PS_TRIM_LOW = 0.025
PS_TRIM_HIGH = 0.975

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def to_naive_utc_day(series):
    """Robustly converts mixed timezones to naive UTC midnight."""
    return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()

def clean_mem():
    """Forces garbage collection."""
    gc.collect()

def calculate_ess(weights):
    """Calculates Kish's Effective Sample Size."""
    if len(weights) == 0: return 0
    return (np.sum(weights) ** 2) / np.sum(weights ** 2)

def calculate_e_value(rr):
    """Calculates E-Value for unmeasured confounding sensitivity."""
    if rr <= 0 or np.isnan(rr):
        return 1.0
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))

# =============================================================================
# 3. SUPER LEARNER AIPW ENGINE
# =============================================================================

def get_super_learner_xgb(n_jobs_inner=1):
    """
    Returns a SuperLearner-style stacking classifier with diverse XGBoost base learners.
    Library includes: GAM-like, Moderate depth, Deep, and Regularized models.
    """
    try:
        import xgboost as xgb
    except ImportError:
        # Fallback to sklearn GradientBoosting if XGBoost not available
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_STATE
        )

    common_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'n_jobs': n_jobs_inner,
        'verbosity': 0
    }

    estimators = [
        ('xgb_gam', xgb.XGBClassifier(max_depth=1, **common_params)),      # GAM-like
        ('xgb_moderate', xgb.XGBClassifier(max_depth=4, **common_params)), # Moderate
        ('xgb_deep', xgb.XGBClassifier(max_depth=8, **common_params)),     # Deep
        ('xgb_reg', xgb.XGBClassifier(max_depth=4, reg_lambda=10, **common_params))  # Regularized
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(solver='lbfgs', fit_intercept=True),
        cv=3,
        n_jobs=1,
        passthrough=False
    )
    return stack

def _fit_super_learner_fold(train_idx, eval_idx, X_sparse, T_full, Y_full):
    """Worker function for a single fold of Cross-Fitting."""
    X_train, X_eval = X_sparse[train_idx], X_sparse[eval_idx]
    T_train, T_eval = T_full[train_idx], T_full[eval_idx]
    Y_train, Y_eval = Y_full[train_idx], Y_full[eval_idx]

    # Propensity Score Model
    sl_ps = get_super_learner_xgb(n_jobs_inner=1)
    sl_ps.fit(X_train, T_train)
    pi_hat = sl_ps.predict_proba(X_eval)[:, 1]
    pi_hat = np.clip(pi_hat, PS_TRIM_LOW, PS_TRIM_HIGH)

    # Outcome Models
    mask0 = (T_train == 0)
    mask1 = (T_train == 1)

    # Mu0 (Outcome if No Contrast)
    sl_mu0 = get_super_learner_xgb(n_jobs_inner=1)
    sl_mu0.fit(X_train[mask0], Y_train[mask0])
    mu0_hat = sl_mu0.predict_proba(X_eval)[:, 1]

    # Mu1 (Outcome if Contrast)
    sl_mu1 = get_super_learner_xgb(n_jobs_inner=1)
    sl_mu1.fit(X_train[mask1], Y_train[mask1])
    mu1_hat = sl_mu1.predict_proba(X_eval)[:, 1]

    # Compute EIF (Efficient Influence Function)
    term1 = mu1_hat - mu0_hat
    term2 = (T_eval * (Y_eval - mu1_hat)) / pi_hat
    term3 = ((1 - T_eval) * (Y_eval - mu0_hat)) / (1 - pi_hat)
    eif_chunk = term1 + term2 - term3

    return eval_idx, mu0_hat, mu1_hat, pi_hat, eif_chunk

def run_cross_fitted_aipw(X_sparse_matrix, T_full, Y_full, n_folds=5):
    """Main driver for K-Fold Cross-Fitting AIPW."""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    n = len(T_full)

    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    pi_hat = np.zeros(n)
    eif_val = np.zeros(n)

    print(f"  Running {n_folds}-Fold Super Learner AIPW in PARALLEL...")

    results = Parallel(n_jobs=n_folds, verbose=10)(
        delayed(_fit_super_learner_fold)(train_idx, eval_idx, X_sparse_matrix, T_full, Y_full)
        for train_idx, eval_idx in kf.split(X_sparse_matrix, T_full)
    )

    for eval_idx, mu0_c, mu1_c, pi_c, eif_c in results:
        mu0_hat[eval_idx] = mu0_c
        mu1_hat[eval_idx] = mu1_c
        pi_hat[eval_idx] = pi_c
        eif_val[eval_idx] = eif_c

    # Statistics & Inference
    ate = np.mean(eif_val)
    se = np.std(eif_val) / np.sqrt(n)
    p_value = 2 * (1 - norm.cdf(np.abs(ate / se))) if se > 0 else 0.0

    risk_1 = np.mean(mu1_hat)
    risk_0 = np.mean(mu0_hat)
    rr = risk_1 / risk_0 if risk_0 > 0 else 0.0

    w = np.where(T_full == 1, 1/pi_hat, 1/(1-pi_hat))
    ess = calculate_ess(w)

    e_calc_rr = 1/rr if (rr < 1 and rr > 0) else rr
    e_val = calculate_e_value(e_calc_rr) if not np.isnan(rr) else 1.0

    stats = {
        'ATE': ate, 'SE': se, 'P_Value': p_value,
        'Risk_1': risk_1, 'Risk_0': risk_0,
        'RR': rr, 'ESS': ess, 'E_Value': e_val,
        'CI_Lower': ate - 1.96*se, 'CI_Upper': ate + 1.96*se
    }

    predictions = {'mu0': mu0_hat, 'mu1': mu1_hat, 'pi': pi_hat, 'eif': eif_val}
    return stats, predictions

# =============================================================================
# 4. POLICY EVALUATION
# =============================================================================

def policy_current(df):
    """Current Practice: The observed decision."""
    return df['contrast_received'].values

def policy_always(df):
    """Always give contrast (100% contrast use)."""
    return np.ones(len(df), dtype=int)

def policy_never(df):
    """Never give contrast (0% contrast use)."""
    return np.zeros(len(df), dtype=int)

def policy_egfr_rule_30(df):
    """Withhold contrast if eGFR < 30, otherwise give."""
    return (df['egfr_cat'] != 0).astype(int).values

def policy_egfr_rule_45(df):
    """Withhold contrast if eGFR < 45, otherwise give."""
    return (~df['egfr_cat'].isin([0, 1])).astype(int).values

def evaluate_policies(df_final, T_vec, Y_vec, preds):
    """Evaluate counterfactual treatment policies."""
    policies = {
        'Current Practice': policy_current,
        'Always Contrast (100%)': policy_always,
        'Never Contrast (0%)': policy_never,
        'eGFR Rule: withhold if <30': policy_egfr_rule_30,
        'eGFR Rule: withhold if <45': policy_egfr_rule_45,
    }

    mu1, mu0, pi = preds['mu1'], preds['mu0'], preds['pi']

    # DR estimators for potential outcomes
    gamma_1 = mu1 + (T_vec / pi) * (Y_vec - mu1)
    gamma_0 = mu0 + ((1 - T_vec) / (1 - pi)) * (Y_vec - mu0)

    results = []
    for name, func in policies.items():
        d_vec = func(df_final)
        psi_i = d_vec * gamma_1 + (1 - d_vec) * gamma_0

        results.append({
            'Policy': name,
            'Risk': np.mean(psi_i),
            'Risk_SE': np.std(psi_i) / np.sqrt(len(psi_i)),
            'Withholding': np.mean(1 - d_vec)
        })

    return pd.DataFrame(results)

# =============================================================================
# 5. MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("AKI-CONTRAST TRIAL EMULATION")
    print("Contrast-Enhanced vs Non-Contrast Imaging and AKI Risk")
    print("=" * 60)

    # Load checkpoint data
    print("\nLoading checkpoint data...")
    try:
        df_final = pd.read_parquet("df_final.parquet")
        X_final = sp.load_npz("X_final.npz")
        print(f"Data loaded: Cohort N={len(df_final)}, Features={X_final.shape[1]}")
    except FileNotFoundError:
        print("ERROR: Checkpoint files not found.")
        print("Required: df_final.parquet, X_final.npz")
        return

    # Define treatment and outcome
    T_vec = df_final['contrast_received'].values
    col_date = 'date_AKI_30'
    Y_vec = ((df_final[col_date] - df_final['index_date']).dt.days <= 30).astype(int).values

    print(f"\nCohort Summary:")
    print(f"  - Total N: {len(df_final)}")
    print(f"  - Contrast (T=1): {T_vec.sum()}")
    print(f"  - No Contrast (T=0): {len(T_vec) - T_vec.sum()}")
    print(f"  - AKI Events: {Y_vec.sum()} ({Y_vec.mean()*100:.2f}%)")

    # Run AIPW Analysis
    print("\n" + "-" * 60)
    print("RUNNING CROSS-FITTED AIPW ANALYSIS")
    print("-" * 60)

    stats, preds = run_cross_fitted_aipw(X_final, T_vec, Y_vec, n_folds=N_FOLDS)

    # Print Primary Results
    print("\n" + "=" * 60)
    print("PRIMARY RESULTS: CONTRAST vs NO-CONTRAST")
    print("=" * 60)
    print(f"ATE (Risk Difference): {stats['ATE']:.4f} [{stats['CI_Lower']:.4f}, {stats['CI_Upper']:.4f}]")
    print(f"P-Value: {stats['P_Value']:.4g}")
    print(f"Risk if Contrast (E[Y|do(T=1)]): {stats['Risk_1']:.4f}")
    print(f"Risk if No Contrast (E[Y|do(T=0)]): {stats['Risk_0']:.4f}")
    print(f"Risk Ratio: {stats['RR']:.3f}")
    print(f"E-Value: {stats['E_Value']:.2f}")
    print(f"Effective Sample Size: {stats['ESS']:.1f}")

    # Policy Evaluation
    print("\n" + "-" * 60)
    print("POLICY EVALUATION")
    print("-" * 60)

    df_policies = evaluate_policies(df_final, T_vec, Y_vec, preds)
    print(df_policies.to_string(index=False))

    # Regulatory Conclusion
    print("\n" + "=" * 60)
    print("REGULATORY CONCLUSION")
    print("=" * 60)

    if stats['ATE'] < 0 and stats['P_Value'] < 0.05:
        print("FINDING: CONTRAST IS SAFE")
        print(f"Contrast-enhanced imaging REDUCES AKI risk by {abs(stats['ATE'])*100:.1f} percentage points.")
        print(f"Risk Ratio of {stats['RR']:.2f} indicates ~{(1-stats['RR'])*100:.0f}% relative risk reduction.")
        print(f"E-value of {stats['E_Value']:.2f} indicates robustness to unmeasured confounding.")
    elif stats['P_Value'] >= 0.05:
        print("FINDING: NO SIGNIFICANT DIFFERENCE")
        print("Evidence does not support a difference between contrast and no-contrast imaging.")
    else:
        print("FINDING: POTENTIAL HARM")
        print("Further investigation required.")

    print("=" * 60)

    return stats, preds, df_policies

if __name__ == "__main__":
    main()
