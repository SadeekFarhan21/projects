"""
LEAP Funding Model - Pipe Count & Consumer Adoption Simulation
==============================================================
1. Count pipes needing replacement from real system data
2. Simulate consumer adoption/replacement timing
3. Estimate total upfront funding needed (2026-2037)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

np.random.seed(42)

# ============================================================
# PART 1: COUNT PIPES NEEDING REPLACEMENT
# ============================================================

print("=" * 70)
print("PART 1: PIPE INVENTORY - WHAT NEEDS REPLACING?")
print("=" * 70)

# From System Materials data
system_materials = {
    # (Utility Material, Customer Material): (Count, Whole Line Designation)
    ("Galvanized", "Galvanized"): (1020, "Galvanized"),
    ("Galvanized", "Non-Lead - Copper"): (1248, "Galvanized"),
    ("Galvanized", "Non-Lead - Other"): (1, "Galvanized"),
    ("Galvanized", "Non-Lead - Plastic"): (702, "Galvanized"),
    ("Galvanized", "Unknown"): (2425, "Galvanized"),
    ("Non-Lead - Copper", "Galvanized"): (7253, "Galvanized"),
    ("Non-Lead - Other", "Galvanized"): (40, "Galvanized"),
    ("Non-Lead - Plastic", "Galvanized"): (112, "Galvanized"),
    ("Unknown", "Galvanized"): (1843, "Galvanized"),
    ("Lead", "Galvanized"): (4086, "Lead"),
    ("Lead", "Non-Lead - Copper"): (5107, "Lead"),
    ("Lead", "Non-Lead - Other"): (2, "Lead"),
    ("Lead", "Non-Lead - Plastic"): (3134, "Lead"),
    ("Lead", "Unknown"): (11672, "Lead"),
    ("Non-Lead - Copper", "Non-Lead - Copper"): (32619, "Non-Lead"),
    ("Non-Lead - Copper", "Non-Lead - Other"): (16330, "Non-Lead"),
    ("Non-Lead - Copper", "Non-Lead - Plastic"): (19765, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Copper"): (5597, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Other"): (32820, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Plastic"): (8401, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Copper"): (2132, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Other"): (55183, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Plastic"): (21882, "Non-Lead"),
    ("Non-Lead - Copper", "Unknown"): (27941, "Unknown"),
    ("Non-Lead - Other", "Unknown"): (119, "Unknown"),
    ("Non-Lead - Plastic", "Unknown"): (1783, "Unknown"),
    ("Unknown", "Non-Lead - Copper"): (7230, "Unknown"),
    ("Unknown", "Non-Lead - Other"): (35, "Unknown"),
    ("Unknown", "Non-Lead - Plastic"): (1874, "Unknown"),
    ("Unknown", "Unknown"): (16033, "Unknown"),
}

# Aggregate by designation
designation_counts = defaultdict(int)
for (util_mat, cust_mat), (count, designation) in system_materials.items():
    designation_counts[designation] += count

total_lines = sum(designation_counts.values())

print("\nService Line Inventory by Designation:")
print("-" * 45)
for designation, count in sorted(designation_counts.items()):
    pct = count / total_lines * 100
    print(f"  {designation:15s}: {count:>7,d} lines ({pct:5.1f}%)")
print(f"  {'TOTAL':15s}: {total_lines:>7,d} lines")

# Lines that MUST be replaced (federal mandate)
lead_count = designation_counts["Lead"]
galvanized_count = designation_counts["Galvanized"]
must_replace = lead_count + galvanized_count

print(f"\n*** MANDATORY REPLACEMENT (Lead + Galvanized): {must_replace:,d} lines ***")

# Unknown lines - estimate what fraction are lead/galvanized
unknown_count = designation_counts["Unknown"]
known_total = total_lines - unknown_count
lead_frac_of_known = lead_count / known_total
galv_frac_of_known = galvanized_count / known_total
hazardous_frac = (lead_count + galvanized_count) / known_total

print(f"\nUnknown lines: {unknown_count:,d}")
print(f"  Among KNOWN lines, hazardous rate = {hazardous_frac:.1%}")
print(f"  Estimated hazardous unknowns = {int(unknown_count * hazardous_frac):,d}")

estimated_total_replacements = must_replace + int(unknown_count * hazardous_frac)
print(f"\n*** ESTIMATED TOTAL REPLACEMENTS NEEDED: {estimated_total_replacements:,d} lines ***")

# ============================================================
# PART 2: COLUMBUS v4 TEST DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 2: COLUMBUS v4 TEST DATA PROFILE")
print("=" * 70)

df = pd.read_csv("/Users/farhan/Documents/GitHub/quantathon-2026/dataset/Columbus v4.csv", skiprows=1)

print(f"\nTotal properties: {len(df):,d}")
print(f"\nMaterial distribution:")
print(df['line_material'].value_counts())

# Properties needing replacement
needs_replacement = df[df['line_material'].isin(['lead', 'galvanized'])]
print(f"\nProperties needing replacement: {len(needs_replacement):,d} ({len(needs_replacement)/len(df):.1%})")
print(f"  Lead: {len(df[df['line_material']=='lead']):,d}")
print(f"  Galvanized: {len(df[df['line_material']=='galvanized']):,d}")

print(f"\nReplacement cost stats (all properties):")
print(df['estimated_replacement_cost'].describe())

print(f"\nReplacement cost stats (lead/galvanized only):")
print(needs_replacement['estimated_replacement_cost'].describe())

# LEAP loan cap is $10,000
loan_cap = 10000
needs_replacement = needs_replacement.copy()
needs_replacement['loan_amount'] = needs_replacement['estimated_replacement_cost'].clip(upper=loan_cap)
avg_loan = needs_replacement['loan_amount'].mean()
print(f"\nAverage LEAP loan amount (capped at $10K): ${avg_loan:,.2f}")

print(f"\nProperty type distribution (needing replacement):")
print(needs_replacement['property_type'].value_counts())

print(f"\nOwner-occupied rate (needing replacement): {needs_replacement['owner_occupied'].mean():.1%}")

print(f"\nLead risk score distribution (needing replacement):")
print(needs_replacement['lead_risk'].describe())

print(f"\nYear built distribution (needing replacement):")
print(needs_replacement['year_built'].describe())

print(f"\nOwnership length distribution (needing replacement):")
print(needs_replacement['ownership_length'].describe())

# ============================================================
# PART 3: LEAP PROGRAM PARTICIPATION FUNNEL
# ============================================================

print("\n" + "=" * 70)
print("PART 3: LEAP PARTICIPATION FUNNEL & QUOTE ANALYSIS")
print("=" * 70)

# Participation data
funnel = {
    "Interested applicants": 127,
    "Applications sent": 90,
    "Applied": 82,
    "Dropped out after applying": 18,
    "Rejected": 2,
    "Completed loans": 42,
    "In-progress loans": 20,
    "Proactive loans": 31,
    "Leak loans": 59,
}

print("\nLEAP Participation Funnel:")
print("-" * 50)
for k, v in funnel.items():
    print(f"  {k:40s}: {v:>5d}")

# Conversion rates
interest_to_apply = 82 / 127
apply_to_proceed = (82 - 18 - 2) / 82
overall_conversion = 42 / 127

print(f"\nConversion rates:")
print(f"  Interest → Application: {interest_to_apply:.1%}")
print(f"  Application → Proceed: {apply_to_proceed:.1%}")
print(f"  Overall (Interest → Completed): {overall_conversion:.1%}")

# Quote analysis
quotes_chosen = [17410, 12610, 6250, 8510, 8840, 7000, 9500, 6900, 5300, 6800,
                 6250, 8200, 10000, 9000, 4800, 7500, 7600, 8000, 9600, 9200,
                 6445, 6500, 5150, 9100, 9600, 6800, 6200, 9095, 9685, 6500,
                 7500, 7500, 5200, 6950, 6750, 10185, 6400, 6800, 6800, 6800,
                 6800, 5200, 6800, 6800, 5327.84, 6100, 7988.4, 7121, 4250,
                 5488, 9070, 7200, 5800, 6875, 8550, 7650, 10063.93, 10536.74,
                 9955, 5500, 8283.8, 6500, 6660, 5500]

loan_amounts_actual = [min(q, 10000) for q in quotes_chosen]

print(f"\nContractor Quote Analysis (chosen quotes):")
print(f"  Number of quotes: {len(quotes_chosen)}")
print(f"  Mean chosen quote: ${np.mean(quotes_chosen):,.2f}")
print(f"  Median chosen quote: ${np.median(quotes_chosen):,.2f}")
print(f"  Min: ${min(quotes_chosen):,.2f}")
print(f"  Max: ${max(quotes_chosen):,.2f}")
print(f"\n  Mean LEAP loan (capped at $10K): ${np.mean(loan_amounts_actual):,.2f}")
print(f"  Median LEAP loan: ${np.median(loan_amounts_actual):,.2f}")

# ============================================================
# PART 4: CONSUMER ADOPTION SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("PART 4: CONSUMER ADOPTION SIMULATION (2026-2037)")
print("=" * 70)

# ---- Key Parameters ----
# Total lines needing replacement in Columbus system
TOTAL_LINES_NEEDING_REPLACEMENT = estimated_total_replacements  # from Part 1

# From Columbus v4 test data proportions
LEAD_FRACTION = len(df[df['line_material'] == 'lead']) / len(needs_replacement)
GALV_FRACTION = len(df[df['line_material'] == 'galvanized']) / len(needs_replacement)

# Annual home sale rate: ~5-7% nationally, Columbus ~6%
ANNUAL_SALE_RATE = 0.06

# Inheritance rate: ~1.5% of properties per year transfer via inheritance
ANNUAL_INHERITANCE_RATE = 0.015

# Equity extraction (refinancing/HELOC) rate: ~3-4% of homeowners per year
ANNUAL_EQUITY_EXTRACTION_RATE = 0.035

# LEAP participation rate (from funnel data)
# 127 interested out of ~38,645 needing replacement = very low initial uptake
# But this is early days. We model adoption with an S-curve as awareness grows
# and deadline pressure increases.

# Average loan amount (from actual quotes, capped at $10K)
AVG_LOAN_AMOUNT = np.mean(loan_amounts_actual)

# Adoption model parameters
# As 2037 deadline approaches, adoption accelerates
YEARS = list(range(2026, 2038))  # 2026-2037
N_YEARS = len(YEARS)

# S-curve adoption: slow start, acceleration, saturation
# By mandate, 100% must be replaced by 2037
def adoption_schedule(total_lines, base_rate=0.05, acceleration=0.25):
    """
    Generate year-by-year adoption rates.
    Early years: voluntary, slow adoption.
    Later years: mandatory pressure increases adoption.
    """
    annual_fractions = []
    cumulative = 0
    for yr_idx in range(N_YEARS):
        years_left = N_YEARS - yr_idx
        remaining = total_lines - cumulative

        if years_left <= 1:
            # Final year: everyone remaining MUST replace
            frac = remaining
        else:
            # S-curve: rate increases as deadline nears
            urgency = 1 + acceleration * (yr_idx / N_YEARS) ** 2 * 10
            voluntary_rate = base_rate * urgency
            # Add forced compliance near deadline
            if years_left <= 3:
                voluntary_rate += (1 - voluntary_rate) * (0.3 * (4 - years_left) / 3)
            frac = min(remaining, int(remaining * voluntary_rate))

        annual_fractions.append(frac)
        cumulative += frac

    return annual_fractions

print(f"\nKey assumptions:")
print(f"  Total lines needing replacement: {TOTAL_LINES_NEEDING_REPLACEMENT:,d}")
print(f"  Average LEAP loan amount: ${AVG_LOAN_AMOUNT:,.2f}")
print(f"  Annual home sale rate: {ANNUAL_SALE_RATE:.1%}")
print(f"  Annual inheritance rate: {ANNUAL_INHERITANCE_RATE:.1%}")
print(f"  Annual equity extraction rate: {ANNUAL_EQUITY_EXTRACTION_RATE:.1%}")
print(f"  LEAP conversion rate (interest → loan): {overall_conversion:.1%}")

# ---- Monte Carlo Simulation (Cohort-based, vectorized) ----
N_SIMS = 10000

print(f"\nRunning {N_SIMS:,d} Monte Carlo simulations (cohort-based)...")

# Store results
annual_disbursements = np.zeros((N_SIMS, N_YEARS))
annual_repayments = np.zeros((N_SIMS, N_YEARS))
cumulative_fund_needed = np.zeros((N_SIMS, N_YEARS))

# Pre-generate random parameters for all simulations
total_lines_all = np.clip(
    np.random.normal(TOTAL_LINES_NEEDING_REPLACEMENT, TOTAL_LINES_NEEDING_REPLACEMENT * 0.05, N_SIMS),
    must_replace, None
).astype(int)

avg_loan_all = np.random.normal(AVG_LOAN_AMOUNT, 500, N_SIMS)
sale_rate_all = np.clip(np.random.normal(ANNUAL_SALE_RATE, 0.01, N_SIMS), 0.02, 0.12)
inherit_rate_all = np.clip(np.random.normal(ANNUAL_INHERITANCE_RATE, 0.005, N_SIMS), 0.005, 0.03)
equity_rate_all = np.clip(np.random.normal(ANNUAL_EQUITY_EXTRACTION_RATE, 0.01, N_SIMS), 0.01, 0.08)
base_rate_all = np.clip(np.random.normal(0.05, 0.01, N_SIMS), 0.02, 0.10)
accel_all = np.clip(np.random.normal(0.25, 0.05, N_SIMS), 0.10, 0.40)

for sim in range(N_SIMS):
    total_lines = total_lines_all[sim]
    avg_loan = avg_loan_all[sim]
    sale_rate = sale_rate_all[sim]
    inherit_rate = inherit_rate_all[sim]
    equity_rate = equity_rate_all[sim]

    # Repayment probability per year = sale + equity extraction
    repay_rate = sale_rate + equity_rate  # ~9.5% annual repayment probability

    replacements_per_year = adoption_schedule(total_lines, base_rate_all[sim], accel_all[sim])

    # Track cohorts: outstanding[yr_idx] = $ still outstanding from loans issued in year yr_idx
    cohort_outstanding = np.zeros(N_YEARS)

    for yr_idx in range(N_YEARS):
        n_new = replacements_per_year[yr_idx]
        disbursed = n_new * avg_loan
        annual_disbursements[sim, yr_idx] = disbursed
        cohort_outstanding[yr_idx] += disbursed

        # Apply repayments to ALL existing cohorts
        total_repaid = 0
        for c in range(yr_idx + 1):
            repaid = cohort_outstanding[c] * repay_rate
            total_repaid += repaid
            cohort_outstanding[c] -= repaid

        annual_repayments[sim, yr_idx] = total_repaid

    # Cumulative net funding
    cumulative_fund_needed[sim] = np.cumsum(annual_disbursements[sim] - annual_repayments[sim])

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS: FUNDING REQUIREMENTS")
print("=" * 70)

# Total disbursements across all years
total_disbursements = annual_disbursements.sum(axis=1)
total_repayments = annual_repayments.sum(axis=1)

# Peak cumulative funding = maximum outstanding at any point
peak_funding = cumulative_fund_needed.max(axis=1)

# Grant model = total disbursements (no repayments ever)
grant_model_cost = total_disbursements

print("\n--- LOAN-BASED MODEL ---")
print(f"Total disbursements (mean): ${total_disbursements.mean():,.0f}")
print(f"Total repayments by 2037 (mean): ${total_repayments.mean():,.0f}")
print(f"Peak outstanding fund needed (mean): ${peak_funding.mean():,.0f}")
print(f"Peak outstanding fund (median): ${np.median(peak_funding):,.0f}")
print(f"Peak outstanding fund (95th percentile): ${np.percentile(peak_funding, 95):,.0f}")
print(f"\n  *** 95% confidence funding level: ${np.percentile(peak_funding, 95):,.0f} ***")

print("\n--- GRANT-BASED MODEL ---")
print(f"Total cost (mean): ${grant_model_cost.mean():,.0f}")
print(f"Total cost (median): ${np.median(grant_model_cost):,.0f}")
print(f"Total cost (95th percentile): ${np.percentile(grant_model_cost, 95):,.0f}")
print(f"\n  *** 95% confidence funding level: ${np.percentile(grant_model_cost, 95):,.0f} ***")

print(f"\n--- SAVINGS FROM LOAN vs GRANT ---")
savings = grant_model_cost - peak_funding
print(f"Mean savings with loan model: ${savings.mean():,.0f}")
print(f"Loan model is {savings.mean() / grant_model_cost.mean():.1%} cheaper than grant model")

# ---- Year-by-year breakdown ----
print("\n--- YEAR-BY-YEAR PROJECTIONS (Mean) ---")
print(f"{'Year':<6} {'Replacements':>13} {'Disbursed':>14} {'Repaid':>12} {'Net Outstanding':>16}")
print("-" * 65)

mean_replacements = adoption_schedule(TOTAL_LINES_NEEDING_REPLACEMENT)
for yr_idx in range(N_YEARS):
    print(f"{YEARS[yr_idx]:<6} {mean_replacements[yr_idx]:>13,d} "
          f"${annual_disbursements[:, yr_idx].mean():>12,.0f} "
          f"${annual_repayments[:, yr_idx].mean():>10,.0f} "
          f"${cumulative_fund_needed[:, yr_idx].mean():>14,.0f}")

# ============================================================
# PART 5: VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("LEAP Funding Model Analysis", fontsize=16, fontweight='bold')

# 1. Pipe Material Distribution (System)
ax = axes[0, 0]
labels = list(designation_counts.keys())
sizes = list(designation_counts.values())
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6']
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('System-Wide Pipe Material Distribution')

# 2. LEAP Participation Funnel
ax = axes[0, 1]
funnel_labels = ['Interested\n(127)', 'App Sent\n(90)', 'Applied\n(82)',
                 'Proceeded\n(62)', 'Completed\n(42)']
funnel_values = [127, 90, 82, 62, 42]
bars = ax.barh(funnel_labels[::-1], funnel_values[::-1], color='#3498db')
ax.set_xlabel('Number of Homeowners')
ax.set_title('LEAP Participation Funnel')
for bar, val in zip(bars, funnel_values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{val}', va='center', fontweight='bold')

# 3. Replacement Cost Distribution (from quotes)
ax = axes[0, 2]
ax.hist(quotes_chosen, bins=15, color='#e67e22', edgecolor='black', alpha=0.8)
ax.axvline(x=10000, color='red', linestyle='--', linewidth=2, label='$10K LEAP Cap')
ax.axvline(x=np.mean(quotes_chosen), color='blue', linestyle='--', linewidth=2,
           label=f'Mean: ${np.mean(quotes_chosen):,.0f}')
ax.set_xlabel('Replacement Cost ($)')
ax.set_ylabel('Frequency')
ax.set_title('Contractor Quote Distribution')
ax.legend()

# 4. Year-by-year funding (loan model)
ax = axes[1, 0]
mean_cum = cumulative_fund_needed.mean(axis=0)
p5 = np.percentile(cumulative_fund_needed, 5, axis=0)
p95 = np.percentile(cumulative_fund_needed, 95, axis=0)
ax.fill_between(YEARS, p5, p95, alpha=0.3, color='#3498db', label='5th-95th percentile')
ax.plot(YEARS, mean_cum, 'b-', linewidth=2, label='Mean')
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative Net Funding ($)')
ax.set_title('Loan Model: Outstanding Fund Over Time')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))

# 5. Loan vs Grant comparison
ax = axes[1, 1]
loan_95 = np.percentile(peak_funding, 95)
grant_95 = np.percentile(grant_model_cost, 95)
bars = ax.bar(['Loan Model\n(Peak Outstanding)', 'Grant Model\n(Total Cost)'],
              [loan_95, grant_95], color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_ylabel('Funding Required ($)')
ax.set_title('95% Confidence Funding Level')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
for bar, val in zip(bars, [loan_95, grant_95]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${val/1e6:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 6. Distribution of total funding needed
ax = axes[1, 2]
ax.hist(peak_funding / 1e6, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black',
        label='Loan Model (Peak)')
ax.hist(grant_model_cost / 1e6, bins=50, color='#e74c3c', alpha=0.5, edgecolor='black',
        label='Grant Model (Total)')
ax.axvline(x=np.percentile(peak_funding, 95)/1e6, color='green', linestyle='--',
           linewidth=2, label=f'Loan 95%: ${np.percentile(peak_funding, 95)/1e6:.1f}M')
ax.axvline(x=np.percentile(grant_model_cost, 95)/1e6, color='red', linestyle='--',
           linewidth=2, label=f'Grant 95%: ${np.percentile(grant_model_cost, 95)/1e6:.1f}M')
ax.set_xlabel('Funding Required ($M)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Funding Requirements')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/farhan/Documents/GitHub/quantathon-2026/analysis/funding_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: analysis/funding_analysis.png")

# ---- Additional: Consumer behavior analysis ----
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Consumer Analytics - Who Replaces and When?", fontsize=14, fontweight='bold')

# Lead risk score vs replacement likelihood
ax = axes2[0]
for mat, color, label in [('lead', '#e74c3c', 'Lead'), ('galvanized', '#f39c12', 'Galvanized'), ('copper', '#2ecc71', 'Copper')]:
    subset = df[df['line_material'] == mat]
    ax.hist(subset['lead_risk'], bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
ax.set_xlabel('Lead Risk Score')
ax.set_ylabel('Number of Properties')
ax.set_title('Lead Risk Score by Material Type')
ax.legend()

# Year built vs material type
ax = axes2[1]
for mat, color, label in [('lead', '#e74c3c', 'Lead'), ('galvanized', '#f39c12', 'Galvanized'), ('copper', '#2ecc71', 'Copper')]:
    subset = df[df['line_material'] == mat]
    ax.hist(subset['year_built'], bins=20, alpha=0.6, color=color, label=label, edgecolor='black')
ax.set_xlabel('Year Built')
ax.set_ylabel('Number of Properties')
ax.set_title('Year Built Distribution by Material')
ax.legend()

# Ownership length distribution (affects sale probability)
ax = axes2[2]
ax.hist(needs_replacement['ownership_length'], bins=30, color='#3498db', edgecolor='black', alpha=0.8)
ax.axvline(x=needs_replacement['ownership_length'].median(), color='red', linestyle='--',
           label=f"Median: {needs_replacement['ownership_length'].median():.0f} yrs")
ax.set_xlabel('Ownership Length (years)')
ax.set_ylabel('Number of Properties')
ax.set_title('Ownership Length (Properties Needing Replacement)')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/farhan/Documents/GitHub/quantathon-2026/analysis/consumer_analytics.png', dpi=150, bbox_inches='tight')
print("Saved: analysis/consumer_analytics.png")

# ---- Insurance trend analysis ----
fig3, ax = plt.subplots(figsize=(10, 5))
dates = ['Oct 2025', 'Nov 2025', 'Jan 2026', 'Feb 2026', 'Mar 2026']
wsl_policies = [830, 1468, 2284, 2701, 2862]
wsl_repairs = [4, 10, 20, 28, 41]
active_customers = [869, 1538, 2389, 2833, 3000]

ax.plot(dates, active_customers, 'b-o', linewidth=2, markersize=8, label='Active Customers')
ax.plot(dates, wsl_policies, 'g-s', linewidth=2, markersize=8, label='WSL Policies')
ax2 = ax.twinx()
ax2.plot(dates, wsl_repairs, 'r-^', linewidth=2, markersize=8, label='WSL Repairs')
ax2.set_ylabel('Number of Repairs', color='red')
ax.set_ylabel('Count')
ax.set_title('Insurance Program Growth (Leading Indicator of LEAP Demand)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig('/Users/farhan/Documents/GitHub/quantathon-2026/analysis/insurance_trends.png', dpi=150, bbox_inches='tight')
print("Saved: analysis/insurance_trends.png")

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)
print(f"""
LEAP Funding Model - Key Findings
----------------------------------
1. PIPE INVENTORY:
   - {lead_count:,d} lead lines + {galvanized_count:,d} galvanized = {must_replace:,d} known replacements
   - ~{int(unknown_count * hazardous_frac):,d} additional from {unknown_count:,d} unknown lines
   - TOTAL ESTIMATED: {estimated_total_replacements:,d} lines needing replacement

2. COST PER REPLACEMENT:
   - Average contractor quote: ${np.mean(quotes_chosen):,.0f}
   - Average LEAP loan (capped at $10K): ${np.mean(loan_amounts_actual):,.0f}

3. CONSUMER ADOPTION:
   - Current LEAP conversion rate: {overall_conversion:.1%} (interest → completed)
   - Insurance WSL enrollment growing rapidly (830 → 2,862 in 5 months)
   - Adoption expected to accelerate as 2037 deadline approaches

4. FUNDING RECOMMENDATIONS (95% confidence):
   - LOAN MODEL: ${np.percentile(peak_funding, 95)/1e6:.1f}M upfront
     (repayments reduce net cost over time)
   - GRANT MODEL: ${np.percentile(grant_model_cost, 95)/1e6:.1f}M total
     (no repayments ever)
   - Savings from loan model: ~${savings.mean()/1e6:.1f}M ({savings.mean()/grant_model_cost.mean():.1%})
""")
