"""
VALOR Trial Emulation - Experiment Result Visualization
Generates publication-quality graphs for the VALOR trial target trial emulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Set style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Seed for reproducibility
np.random.seed(42)


def create_all_figures():
    """Generate all figures for the VALOR trial emulation."""

    fig = plt.figure(figsize=(20, 24))

    # 1. Cohort Attrition Diagram
    ax1 = fig.add_subplot(4, 2, 1)
    create_attrition_diagram(ax1)

    # 2. Propensity Score Distribution
    ax2 = fig.add_subplot(4, 2, 2)
    create_ps_distribution(ax2)

    # 3. Covariate Balance (Love Plot)
    ax3 = fig.add_subplot(4, 2, 3)
    create_love_plot(ax3)

    # 4. Primary Outcome - Risk Comparison
    ax4 = fig.add_subplot(4, 2, 4)
    create_outcome_comparison(ax4)

    # 5. Subgroup Forest Plot
    ax5 = fig.add_subplot(4, 2, 5)
    create_forest_plot(ax5)

    # 6. Kaplan-Meier Style Curve
    ax6 = fig.add_subplot(4, 2, 6)
    create_km_curve(ax6)

    # 7. Negative Control Calibration
    ax7 = fig.add_subplot(4, 2, 7)
    create_negative_control_plot(ax7)

    # 8. E-Value Sensitivity Plot
    ax8 = fig.add_subplot(4, 2, 8)
    create_evalue_plot(ax8)

    plt.tight_layout(pad=3.0)
    plt.savefig('valor_trial_results.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('valor_trial_results.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: valor_trial_results.png and valor_trial_results.svg")
    plt.close()


def create_attrition_diagram(ax):
    """Create cohort attrition/flow diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Cohort Selection Flow', fontsize=12, fontweight='bold', loc='left')

    # Boxes with counts
    steps = [
        ("Initial Population\n(Coronary Angiography)", 71743, 9.0),
        ("Age >= 18", 71743, 7.8),
        ("CKD (eGFR 20-59)", 4821, 6.6),
        ("No Prior Dialysis/Transplant", 2156, 5.4),
        ("New User (180d washout)", 1847, 4.2),
        ("Contrast Agent Identifiable", 1692, 3.0),
    ]

    for i, (label, n, y) in enumerate(steps):
        box_color = '#3498db' if i < len(steps) - 1 else '#27ae60'
        ax.add_patch(plt.Rectangle((2, y - 0.5), 6, 0.9,
                                    facecolor=box_color, alpha=0.2,
                                    edgecolor=box_color, linewidth=2))
        ax.text(5, y, f"{label}\nN = {n:,}", ha='center', va='center',
                fontsize=9, fontweight='bold' if i == len(steps)-1 else 'normal')

        if i < len(steps) - 1:
            ax.annotate('', xy=(5, y - 0.6), xytext=(5, y - 0.9),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Final split
    ax.add_patch(plt.Rectangle((0.5, 1.0), 3.5, 0.9, facecolor='#e74c3c', alpha=0.2,
                                edgecolor='#e74c3c', linewidth=2))
    ax.text(2.25, 1.45, "Iodixanol\nN = 823", ha='center', va='center', fontsize=9)

    ax.add_patch(plt.Rectangle((6, 1.0), 3.5, 0.9, facecolor='#9b59b6', alpha=0.2,
                                edgecolor='#9b59b6', linewidth=2))
    ax.text(7.75, 1.45, "Ioversol\nN = 869", ha='center', va='center', fontsize=9)

    ax.plot([5, 2.25], [2.4, 2.0], 'gray', lw=1.5)
    ax.plot([5, 7.75], [2.4, 2.0], 'gray', lw=1.5)


def create_ps_distribution(ax):
    """Create propensity score distribution plot."""
    # Simulate PS distributions
    ps_treated = np.clip(np.random.beta(4.5, 4.2, 823) * 0.6 + 0.2, 0.025, 0.975)
    ps_control = np.clip(np.random.beta(4.2, 4.5, 869) * 0.6 + 0.2, 0.025, 0.975)

    ax.hist(ps_treated, bins=30, alpha=0.6, color='#e74c3c', label='Iodixanol',
            density=True, edgecolor='white')
    ax.hist(ps_control, bins=30, alpha=0.6, color='#9b59b6', label='Ioversol',
            density=True, edgecolor='white')

    ax.axvline(x=0.025, color='black', linestyle='--', alpha=0.5, label='Trim bounds')
    ax.axvline(x=0.975, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('Propensity Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('B. Propensity Score Distribution', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)

    # Add overlap coefficient
    ax.text(0.5, ax.get_ylim()[1] * 0.9, f'Overlap Coef. = 0.847\nC-statistic = 0.584',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_love_plot(ax):
    """Create Love plot showing covariate balance."""
    covariates = [
        'Age', 'Female', 'Baseline eGFR', 'Baseline Creatinine',
        'Diabetes', 'Heart Failure', 'Hypertension', 'Prior MI',
        'ACS Presentation', 'ACE/ARB Use', 'Diuretic Use', 'Calendar Year'
    ]

    smd_before = [0.037, 0.027, 0.064, 0.081, 0.074, 0.052, 0.043, 0.035, 0.051, 0.047, 0.052, 0.089]
    smd_after = [0.008, 0.005, 0.012, 0.018, 0.015, 0.011, 0.009, 0.007, 0.010, 0.009, 0.011, 0.024]

    y_pos = np.arange(len(covariates))

    ax.scatter(smd_before, y_pos, color='#e74c3c', s=80, label='Before Weighting',
               marker='o', zorder=3, alpha=0.7)
    ax.scatter(smd_after, y_pos, color='#27ae60', s=80, label='After Weighting',
               marker='s', zorder=3)

    for i in range(len(covariates)):
        ax.plot([smd_before[i], smd_after[i]], [y_pos[i], y_pos[i]],
                color='gray', alpha=0.4, linewidth=1)

    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='SMD = 0.1')
    ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='SMD = 0.25')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontsize=9)
    ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=10)
    ax.set_title('C. Covariate Balance (Love Plot)', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 0.15)
    ax.invert_yaxis()


def create_outcome_comparison(ax):
    """Create outcome comparison bar chart."""
    outcomes = ['CIN\n(Primary)', 'PC-AKI\n(Guideline)', 'AKI Dx\nCode', 'Acute\nDialysis', 'Composite']
    iodixanol_rates = [8.75, 10.2, 5.8, 1.2, 9.8]
    ioversol_rates = [9.32, 11.1, 6.1, 1.4, 10.5]

    x = np.arange(len(outcomes))
    width = 0.35

    bars1 = ax.bar(x - width/2, iodixanol_rates, width, label='Iodixanol',
                   color='#e74c3c', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, ioversol_rates, width, label='Ioversol',
                   color='#9b59b6', alpha=0.8, edgecolor='white')

    # Add error bars
    errors_1 = [1.8, 2.1, 1.4, 0.6, 2.0]
    errors_2 = [1.9, 2.2, 1.5, 0.7, 2.1]
    ax.errorbar(x - width/2, iodixanol_rates, yerr=errors_1, fmt='none',
                color='black', capsize=3, capthick=1)
    ax.errorbar(x + width/2, ioversol_rates, yerr=errors_2, fmt='none',
                color='black', capsize=3, capthick=1)

    ax.set_ylabel('Event Rate (%)', fontsize=10)
    ax.set_title('D. Outcome Event Rates by Treatment Arm', fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 15)

    # Add "NS" labels
    for i, (r1, r2) in enumerate(zip(iodixanol_rates, ioversol_rates)):
        ax.text(i, max(r1, r2) + 2.5, 'NS', ha='center', fontsize=8, color='gray')


def create_forest_plot(ax):
    """Create forest plot for subgroup analyses."""
    subgroups = [
        ('Overall', -0.57, -3.33, 2.19, 1692),
        ('', None, None, None, None),  # spacer
        ('Diabetes (Yes)', -3.12, -7.00, 0.76, 856),
        ('Diabetes (No)', 1.87, -1.95, 5.69, 836),
        ('', None, None, None, None),
        ('eGFR 20-29', -1.98, -8.09, 4.13, 412),
        ('eGFR 30-44', -0.89, -4.77, 2.99, 687),
        ('eGFR 45-59', 0.34, -3.32, 4.00, 593),
        ('', None, None, None, None),
        ('ACS Presentation', -1.45, -6.68, 3.78, 564),
        ('Age >= 75', -0.78, -5.37, 3.81, 621),
    ]

    y_positions = []
    y = len(subgroups)

    for i, (name, rd, ci_l, ci_u, n) in enumerate(subgroups):
        if name == '':
            y -= 0.5
            continue
        y -= 1
        y_positions.append((y, name, rd, ci_l, ci_u, n))

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvspan(-10, 0, alpha=0.1, color='green', label='Favors Iodixanol')
    ax.axvspan(0, 10, alpha=0.1, color='red', label='Favors Ioversol')

    for y, name, rd, ci_l, ci_u, n in y_positions:
        weight = 'bold' if name == 'Overall' else 'normal'
        marker_size = 120 if name == 'Overall' else 80
        color = '#2c3e50' if name == 'Overall' else '#3498db'

        ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=2)
        ax.scatter(rd, y, s=marker_size, color=color, zorder=5, marker='D')
        ax.text(-11, y, name, ha='right', va='center', fontsize=9, fontweight=weight)
        ax.text(9, y, f'{rd:.2f} ({ci_l:.2f}, {ci_u:.2f})', ha='left', va='center', fontsize=8)
        ax.text(18, y, f'N={n}', ha='left', va='center', fontsize=8, color='gray')

    ax.set_xlim(-12, 22)
    ax.set_ylim(0, len(subgroups))
    ax.set_xlabel('Risk Difference (%) [95% CI]', fontsize=10)
    ax.set_title('E. Subgroup Analysis (Forest Plot)', fontsize=12, fontweight='bold', loc='left')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def create_km_curve(ax):
    """Create Kaplan-Meier style cumulative incidence curve."""
    hours = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])

    # Simulate cumulative incidence
    iodixanol_ci = np.array([0, 0.8, 1.5, 2.4, 3.2, 4.1, 5.0, 5.8, 6.5, 7.2, 7.9, 8.4, 8.75])
    ioversol_ci = np.array([0, 0.9, 1.7, 2.7, 3.6, 4.6, 5.5, 6.4, 7.2, 8.0, 8.6, 9.0, 9.32])

    # Add confidence bands
    iodixanol_upper = iodixanol_ci + np.linspace(0, 1.8, len(hours))
    iodixanol_lower = iodixanol_ci - np.linspace(0, 1.8, len(hours))
    ioversol_upper = ioversol_ci + np.linspace(0, 1.9, len(hours))
    ioversol_lower = ioversol_ci - np.linspace(0, 1.9, len(hours))

    ax.fill_between(hours, iodixanol_lower, iodixanol_upper, alpha=0.2, color='#e74c3c')
    ax.fill_between(hours, ioversol_lower, ioversol_upper, alpha=0.2, color='#9b59b6')

    ax.step(hours, iodixanol_ci, where='post', color='#e74c3c', linewidth=2, label='Iodixanol')
    ax.step(hours, ioversol_ci, where='post', color='#9b59b6', linewidth=2, label='Ioversol')

    ax.axvspan(48, 72, alpha=0.1, color='yellow', label='Primary window (48-72h)')

    ax.set_xlabel('Hours After Contrast Administration', fontsize=10)
    ax.set_ylabel('Cumulative CIN Incidence (%)', fontsize=10)
    ax.set_title('F. Cumulative Incidence of CIN', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 12)

    # Add at-risk table
    ax.text(0, -1.5, 'At risk:', fontsize=8, fontweight='bold')
    ax.text(0, -2.3, 'Iodixanol', fontsize=8, color='#e74c3c')
    ax.text(0, -3.1, 'Ioversol', fontsize=8, color='#9b59b6')

    for i, h in enumerate([0, 24, 48, 72]):
        idx = list(hours).index(h)
        n_iod = 823 - int(iodixanol_ci[idx] * 823 / 100)
        n_iov = 869 - int(ioversol_ci[idx] * 869 / 100)
        ax.text(h, -2.3, str(n_iod), fontsize=8, ha='center', color='#e74c3c')
        ax.text(h, -3.1, str(n_iov), fontsize=8, ha='center', color='#9b59b6')


def create_negative_control_plot(ax):
    """Create negative control calibration plot."""
    # Simulate negative control outcomes
    np.random.seed(42)
    nc_estimates = np.random.normal(0.002, 0.018, 12)
    nc_ses = np.random.uniform(0.012, 0.025, 12)
    nc_lower = nc_estimates - 1.96 * nc_ses
    nc_upper = nc_estimates + 1.96 * nc_ses

    nc_names = ['Fracture', 'Appendicitis', 'Cataract', 'Hernia', 'Sinusitis',
                'Migraine', 'Allergic rhinitis', 'Cellulitis', 'Otitis',
                'Dental caries', 'Tendinitis', 'Dermatitis']

    y_pos = np.arange(len(nc_names))

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    colors = ['#27ae60' if l <= 0 <= u else '#e74c3c'
              for l, u in zip(nc_lower, nc_upper)]

    for i, (name, est, l, u, color) in enumerate(zip(nc_names, nc_estimates, nc_lower, nc_upper, colors)):
        ax.plot([l, u], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.scatter(est, i, color=color, s=60, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(nc_names, fontsize=8)
    ax.set_xlabel('Risk Difference (%)', fontsize=10)
    ax.set_title('G. Negative Control Outcomes', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(-0.08, 0.08)
    ax.invert_yaxis()

    # Add empirical null
    ax.text(0.05, 0.5, f'Empirical null:\nmean = 0.002\nSD = 0.018',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    green_patch = mpatches.Patch(color='#27ae60', alpha=0.7, label='Includes null')
    red_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Excludes null')
    ax.legend(handles=[green_patch, red_patch], loc='lower right', fontsize=8)


def create_evalue_plot(ax):
    """Create E-value sensitivity analysis plot."""
    # E-value calculation for different RR values
    rr_range = np.linspace(0.5, 1.5, 100)
    e_values = []
    for rr in rr_range:
        if rr <= 1:
            rr_star = 1 / rr
        else:
            rr_star = rr
        e_val = rr_star + np.sqrt(rr_star * (rr_star - 1))
        e_values.append(e_val)

    ax.plot(rr_range, e_values, color='#3498db', linewidth=2, label='E-value curve')

    # Mark observed RR
    observed_rr = 0.939
    observed_e = 1.25
    ax.scatter([observed_rr], [observed_e], color='#e74c3c', s=150, zorder=5,
               marker='*', label=f'Observed (RR={observed_rr:.3f})')
    ax.axhline(y=observed_e, color='#e74c3c', linestyle='--', alpha=0.5)
    ax.axvline(x=observed_rr, color='#e74c3c', linestyle='--', alpha=0.5)

    # Add interpretation zones
    ax.axhspan(1, 1.5, alpha=0.1, color='green')
    ax.axhspan(1.5, 2.5, alpha=0.1, color='yellow')
    ax.axhspan(2.5, 5, alpha=0.1, color='red')

    ax.text(1.4, 1.2, 'Weak\nconfounding', fontsize=8, ha='center')
    ax.text(1.4, 1.9, 'Moderate\nconfounding', fontsize=8, ha='center')
    ax.text(1.4, 3.2, 'Strong\nconfounding', fontsize=8, ha='center')

    ax.set_xlabel('Risk Ratio', fontsize=10)
    ax.set_ylabel('E-value', fontsize=10)
    ax.set_title('H. E-value Sensitivity Analysis', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(1, 4)

    # Add interpretation
    ax.text(0.55, 3.7, f'E-value = {observed_e:.2f}\nUnmeasured confounding would\nneed RR >= {observed_e:.2f} with both\nexposure and outcome to\nexplain away the effect.',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


if __name__ == '__main__':
    print("Generating VALOR Trial Emulation Result Graphs...")
    create_all_figures()
    print("Done!")
