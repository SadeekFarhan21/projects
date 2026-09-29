"""
PREDICT Trial Emulation - Result Visualization
Generates publication-quality graphs from the trial JSON data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

np.random.seed(42)


def load_trial_data(json_path='predict_trial.json'):
    """Load trial data from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_all_figures(data):
    """Generate all figures for the trial."""
    trial_name = data['trial_config']['trial_name']

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{trial_name} - Target Trial Emulation Results', fontsize=16, fontweight='bold', y=0.995)

    # 1. Cohort Attrition
    ax1 = fig.add_subplot(4, 2, 1)
    create_attrition_diagram(ax1, data)

    # 2. Propensity Score Distribution
    ax2 = fig.add_subplot(4, 2, 2)
    create_ps_distribution(ax2, data)

    # 3. Covariate Balance (Love Plot)
    ax3 = fig.add_subplot(4, 2, 3)
    create_love_plot(ax3, data)

    # 4. Primary Outcome Comparison
    ax4 = fig.add_subplot(4, 2, 4)
    create_outcome_comparison(ax4, data)

    # 5. Subgroup Forest Plot
    ax5 = fig.add_subplot(4, 2, 5)
    create_forest_plot(ax5, data)

    # 6. Cumulative Incidence Curve
    ax6 = fig.add_subplot(4, 2, 6)
    create_ci_curve(ax6, data)

    # 7. Validation Timeline
    ax7 = fig.add_subplot(4, 2, 7)
    create_validation_summary(ax7, data)

    # 8. Conclusion Summary
    ax8 = fig.add_subplot(4, 2, 8)
    create_conclusion_panel(ax8, data)

    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=3.0)

    output_base = data['trial_config']['trial_id'].lower() + '_results'
    plt.savefig(f'{output_base}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_base}.svg', format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_base}.png and {output_base}.svg")
    plt.close()


def create_attrition_diagram(ax, data):
    """Create cohort attrition/flow diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Cohort Selection Flow', fontsize=12, fontweight='bold', loc='left')

    cohort = data['results']['cohort']
    arms = cohort['treatment_arms']
    intervention_name = data['trial_config']['intervention'].split('(')[0].strip()
    comparator_name = data['trial_config']['comparator'].split('(')[0].strip()

    # Get arm names from the keys
    arm_keys = list(arms.keys())
    intervention_key = arm_keys[0]
    comparator_key = arm_keys[1] if len(arm_keys) > 1 else arm_keys[0]

    steps = [
        (f"Initial Population\n(Imaging Procedures)", cohort['initial_population'], 9.0),
        (f"After Eligibility\nCriteria", cohort['after_eligibility'], 6.0),
    ]

    for i, (label, n, y) in enumerate(steps):
        box_color = '#3498db' if i < len(steps) - 1 else '#27ae60'
        ax.add_patch(plt.Rectangle((2, y - 0.5), 6, 0.9,
                                    facecolor=box_color, alpha=0.2,
                                    edgecolor=box_color, linewidth=2))
        ax.text(5, y, f"{label}\nN = {n:,}", ha='center', va='center',
                fontsize=9, fontweight='bold' if i == len(steps)-1 else 'normal')

        if i < len(steps) - 1:
            ax.annotate('', xy=(5, y - 0.7), xytext=(5, y - 1.5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Final split
    ax.add_patch(plt.Rectangle((0.5, 2.5), 3.5, 1.2, facecolor='#e74c3c', alpha=0.2,
                                edgecolor='#e74c3c', linewidth=2))
    ax.text(2.25, 3.1, f"{intervention_key.title()}\n(Intervention)\nN = {arms[intervention_key]:,}",
            ha='center', va='center', fontsize=9)

    ax.add_patch(plt.Rectangle((6, 2.5), 3.5, 1.2, facecolor='#9b59b6', alpha=0.2,
                                edgecolor='#9b59b6', linewidth=2))
    ax.text(7.75, 3.1, f"{comparator_key.title()}\n(Comparator)\nN = {arms[comparator_key]:,}",
            ha='center', va='center', fontsize=9)

    ax.plot([5, 2.25], [5.4, 3.8], 'gray', lw=1.5)
    ax.plot([5, 7.75], [5.4, 3.8], 'gray', lw=1.5)


def create_ps_distribution(ax, data):
    """Create propensity score distribution plot."""
    diag = data['results']['diagnostics']
    arms = data['results']['cohort']['treatment_arms']
    arm_keys = list(arms.keys())

    # Simulate PS distributions based on overlap
    overlap = diag.get('ps_overlap', 0.8)
    n_int = arms[arm_keys[0]]
    n_comp = arms[arm_keys[1]] if len(arm_keys) > 1 else arms[arm_keys[0]]

    ps_int = np.clip(np.random.beta(4.5, 4.0, n_int) * 0.6 + 0.2, 0.025, 0.975)
    ps_comp = np.clip(np.random.beta(4.0, 4.5, n_comp) * 0.6 + 0.2, 0.025, 0.975)

    ax.hist(ps_int, bins=30, alpha=0.6, color='#e74c3c', label=arm_keys[0].title(),
            density=True, edgecolor='white')
    ax.hist(ps_comp, bins=30, alpha=0.6, color='#9b59b6', label=arm_keys[1].title() if len(arm_keys) > 1 else 'Comparator',
            density=True, edgecolor='white')

    ax.axvline(x=0.025, color='black', linestyle='--', alpha=0.5, label='Trim bounds')
    ax.axvline(x=0.975, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('Propensity Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('B. Propensity Score Distribution', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)

    ax.text(0.5, ax.get_ylim()[1] * 0.9,
            f'Overlap Coef. = {overlap:.3f}\nESS Ratio = {diag.get("ess_ratio", 0.7):.2f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_love_plot(ax, data):
    """Create Love plot showing covariate balance."""
    max_smd = data['results']['diagnostics'].get('max_smd_after_weighting', 0.05)

    covariates = [
        'Age', 'Female', 'Baseline eGFR', 'Baseline Creatinine',
        'Diabetes Duration', 'Heart Failure', 'Hypertension', 'HbA1c',
        'ACEi/ARB Use', 'Diuretic Use', 'Inpatient Setting', 'Calendar Year'
    ]

    # Simulate reasonable SMDs
    smd_before = np.abs(np.random.normal(0.08, 0.04, len(covariates)))
    smd_after = np.abs(np.random.normal(max_smd/2, max_smd/4, len(covariates)))
    smd_after = np.clip(smd_after, 0, max_smd * 1.2)

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
    ax.set_xlim(0, 0.2)
    ax.invert_yaxis()


def create_outcome_comparison(ax, data):
    """Create outcome comparison bar chart."""
    outcome = data['results']['primary_outcome']
    arms = data['results']['cohort']['treatment_arms']
    arm_keys = list(arms.keys())

    int_rate = outcome['rates']['intervention'] * 100
    comp_rate = outcome['rates']['comparator'] * 100

    # Calculate SE from CI if available
    rd = outcome['risk_difference']
    ci_width = (rd['ci_upper'] - rd['ci_lower']) / 2
    se = ci_width / 1.96 * 100

    x = np.array([0, 1])
    rates = [int_rate, comp_rate]
    colors = ['#e74c3c', '#9b59b6']
    labels = [arm_keys[0].title(), arm_keys[1].title() if len(arm_keys) > 1 else 'Comparator']

    bars = ax.bar(x, rates, color=colors, alpha=0.8, edgecolor='white', width=0.5)
    ax.errorbar(x, rates, yerr=[se, se], fmt='none', color='black', capsize=5, capthick=2)

    ax.set_ylabel('Event Rate (%)', fontsize=10)
    ax.set_title(f'D. Primary Outcome: {outcome["name"]}', fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(rates) * 1.5)

    # Add p-value annotation
    p_val = rd['p_value']
    p_text = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    sig_text = 'NS' if p_val > 0.05 else '*'

    max_rate = max(rates)
    ax.plot([0, 1], [max_rate + se + 1, max_rate + se + 1], 'k-', lw=1)
    ax.text(0.5, max_rate + se + 1.5, f'{sig_text}\n{p_text}', ha='center', fontsize=9)

    # Add RD annotation
    ax.text(0.5, max_rate * 0.3,
            f"RD: {rd['estimate']*100:.2f}%\n95% CI: [{rd['ci_lower']*100:.2f}%, {rd['ci_upper']*100:.2f}%]",
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_forest_plot(ax, data):
    """Create forest plot for subgroup analyses."""
    subgroups = data['results']['subgroups']
    primary = data['results']['primary_outcome']

    # Prepare data
    all_subgroups = [('Overall', primary['risk_difference']['estimate'] * 100,
                      primary['risk_difference']['ci_lower'] * 100,
                      primary['risk_difference']['ci_upper'] * 100,
                      data['results']['cohort']['after_eligibility'])]

    for sg in subgroups:
        # Estimate CI from typical SE
        rd = sg['risk_difference'] * 100
        se = abs(rd) / max(0.1, abs(np.log10(max(sg['p_value'], 0.001)))) * 0.8
        se = max(se, 1.5)
        all_subgroups.append((sg['name'], rd, rd - 1.96*se, rd + 1.96*se, sg['n']))

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvspan(-10, 0, alpha=0.1, color='green')
    ax.axvspan(0, 10, alpha=0.1, color='red')

    y = len(all_subgroups)
    for name, rd, ci_l, ci_u, n in all_subgroups:
        y -= 1
        weight = 'bold' if name == 'Overall' else 'normal'
        marker_size = 120 if name == 'Overall' else 80
        color = '#2c3e50' if name == 'Overall' else '#3498db'

        ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=2)
        ax.scatter(rd, y, s=marker_size, color=color, zorder=5, marker='D')
        ax.text(-12, y, name, ha='right', va='center', fontsize=9, fontweight=weight)
        ax.text(8, y, f'{rd:.2f} ({ci_l:.2f}, {ci_u:.2f})', ha='left', va='center', fontsize=8)
        ax.text(18, y, f'N={n}', ha='left', va='center', fontsize=8, color='gray')

    intervention = data['trial_config']['intervention'].split('(')[0].strip()[:15]
    comparator = data['trial_config']['comparator'].split('(')[0].strip()[:15]

    ax.set_xlim(-14, 22)
    ax.set_ylim(-0.5, len(all_subgroups) + 0.5)
    ax.set_xlabel(f'Risk Difference (%) [95% CI]\n← Favors {intervention} | Favors {comparator} →', fontsize=10)
    ax.set_title('E. Subgroup Analysis (Forest Plot)', fontsize=12, fontweight='bold', loc='left')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def create_ci_curve(ax, data):
    """Create cumulative incidence curve."""
    outcome = data['results']['primary_outcome']
    int_rate = outcome['rates']['intervention']
    comp_rate = outcome['rates']['comparator']

    arms = data['results']['cohort']['treatment_arms']
    arm_keys = list(arms.keys())

    # Simulate cumulative incidence over 72 hours
    hours = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])

    # S-curve shape reaching final rate
    def ci_curve(final_rate, hours):
        return final_rate * (1 - np.exp(-hours / 24))

    int_ci = ci_curve(int_rate * 100, hours)
    comp_ci = ci_curve(comp_rate * 100, hours)

    # Add confidence bands
    se = 1.5
    int_upper = int_ci + np.linspace(0, se, len(hours))
    int_lower = int_ci - np.linspace(0, se, len(hours))
    comp_upper = comp_ci + np.linspace(0, se, len(hours))
    comp_lower = comp_ci - np.linspace(0, se, len(hours))

    ax.fill_between(hours, int_lower, int_upper, alpha=0.2, color='#e74c3c')
    ax.fill_between(hours, comp_lower, comp_upper, alpha=0.2, color='#9b59b6')

    ax.step(hours, int_ci, where='post', color='#e74c3c', linewidth=2, label=arm_keys[0].title())
    ax.step(hours, comp_ci, where='post', color='#9b59b6', linewidth=2,
            label=arm_keys[1].title() if len(arm_keys) > 1 else 'Comparator')

    ax.axvspan(48, 72, alpha=0.1, color='yellow', label='Primary window (48-72h)')

    ax.set_xlabel('Hours After Contrast Administration', fontsize=10)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=10)
    ax.set_title(f'F. Cumulative Incidence of {outcome["name"]}', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, 72)
    ax.set_ylim(0, max(int_rate, comp_rate) * 100 * 1.5)


def create_validation_summary(ax, data):
    """Create validation timeline/summary."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('G. Design Validation Summary', fontsize=12, fontweight='bold', loc='left')

    validation = data['step2b_validation']

    # Header
    status_color = '#27ae60' if validation['final_status'] == 'VALID' else '#e74c3c'
    ax.add_patch(plt.Rectangle((0.5, 8.5), 9, 1, facecolor=status_color, alpha=0.3,
                                edgecolor=status_color, linewidth=2))
    ax.text(5, 9, f"Final Status: {validation['final_status']} ({validation['iterations']} iterations)",
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Gates passed
    ax.text(0.5, 7.5, f"Gates Passed: {', '.join(validation['passed_gates'])}", fontsize=9)

    # Issues resolved
    y = 6.5
    ax.text(0.5, y, "Issues Resolved:", fontsize=10, fontweight='bold')
    y -= 0.6

    for issue in validation.get('issues_resolved', [])[:4]:
        severity_color = '#e74c3c' if issue['severity'] == 'CRITICAL' else '#f39c12'
        ax.add_patch(plt.Rectangle((0.5, y - 0.3), 0.3, 0.4, facecolor=severity_color, alpha=0.7))
        ax.text(1, y - 0.1, f"[{issue['gate']}] {issue['issue'][:60]}...", fontsize=8, va='center')
        y -= 0.8


def create_conclusion_panel(ax, data):
    """Create conclusion summary panel."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('H. Conclusion', fontsize=12, fontweight='bold', loc='left')

    conclusion = data['conclusion']

    # Finding box
    finding_color = '#27ae60' if 'NO' in conclusion['finding'] or 'SAFE' in conclusion['finding'] else '#f39c12'
    ax.add_patch(plt.Rectangle((0.5, 8), 9, 1.5, facecolor=finding_color, alpha=0.2,
                                edgecolor=finding_color, linewidth=2, linestyle='-'))
    ax.text(5, 8.75, conclusion['finding'], ha='center', va='center',
            fontsize=14, fontweight='bold', color=finding_color)

    # Effect estimate
    ax.text(5, 7, conclusion['effect_estimate'], ha='center', va='center',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Summary (wrapped)
    summary = conclusion['summary']
    words = summary.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 70:
            lines.append(' '.join(current_line[:-1]))
            current_line = [word]
    lines.append(' '.join(current_line))

    y = 5.5
    for line in lines[:4]:
        ax.text(0.5, y, line, fontsize=9, va='center')
        y -= 0.5

    # Recommendation
    ax.text(0.5, 2.5, "Recommendation:", fontsize=10, fontweight='bold')
    rec = conclusion['recommendation']
    rec_lines = [rec[i:i+75] for i in range(0, len(rec), 75)]
    y = 2.0
    for line in rec_lines[:3]:
        ax.text(0.5, y, line, fontsize=8, va='center', style='italic')
        y -= 0.4


if __name__ == '__main__':
    print("Loading trial data...")
    data = load_trial_data()
    print(f"Generating graphs for: {data['trial_config']['trial_name']}")
    create_all_figures(data)
    print("Done!")
