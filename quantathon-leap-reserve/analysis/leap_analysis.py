"""
LEAP Funding Analysis
=====================
Focused on the real LEAP data (CWP v2 training set):
  - System Materials: pipe inventory & what needs replacing
  - LEAP Participation: adoption funnel
  - LEAP Quotes: actual replacement costs & loan amounts
  - Insurance: WSL demand growth signal

Goal: Estimate 2026 upfront funding for loan-based and grant-based models
      with 95% confidence intervals via Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)
OUT = "/Users/farhan/Documents/GitHub/quantathon-2026/analysis"

# ============================================================
# 1. PIPE INVENTORY — How many lines must be replaced?
# ============================================================

# From CWP - LEAP Data v2 - System Materials
materials = [
    # (Utility, Customer, Count, Designation)
    ("Galvanized", "Galvanized", 1020, "Galvanized"),
    ("Galvanized", "Non-Lead - Copper", 1248, "Galvanized"),
    ("Galvanized", "Non-Lead - Other", 1, "Galvanized"),
    ("Galvanized", "Non-Lead - Plastic", 702, "Galvanized"),
    ("Galvanized", "Unknown", 2425, "Galvanized"),
    ("Non-Lead - Copper", "Galvanized", 7253, "Galvanized"),
    ("Non-Lead - Other", "Galvanized", 40, "Galvanized"),
    ("Non-Lead - Plastic", "Galvanized", 112, "Galvanized"),
    ("Unknown", "Galvanized", 1843, "Galvanized"),
    ("Lead", "Galvanized", 4086, "Lead"),
    ("Lead", "Non-Lead - Copper", 5107, "Lead"),
    ("Lead", "Non-Lead - Other", 2, "Lead"),
    ("Lead", "Non-Lead - Plastic", 3134, "Lead"),
    ("Lead", "Unknown", 11672, "Lead"),
    ("Non-Lead - Copper", "Non-Lead - Copper", 32619, "Non-Lead"),
    ("Non-Lead - Copper", "Non-Lead - Other", 16330, "Non-Lead"),
    ("Non-Lead - Copper", "Non-Lead - Plastic", 19765, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Copper", 5597, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Other", 32820, "Non-Lead"),
    ("Non-Lead - Other", "Non-Lead - Plastic", 8401, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Copper", 2132, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Other", 55183, "Non-Lead"),
    ("Non-Lead - Plastic", "Non-Lead - Plastic", 21882, "Non-Lead"),
    ("Non-Lead - Copper", "Unknown", 27941, "Unknown"),
    ("Non-Lead - Other", "Unknown", 119, "Unknown"),
    ("Non-Lead - Plastic", "Unknown", 1783, "Unknown"),
    ("Unknown", "Non-Lead - Copper", 7230, "Unknown"),
    ("Unknown", "Non-Lead - Other", 35, "Unknown"),
    ("Unknown", "Non-Lead - Plastic", 1874, "Unknown"),
    ("Unknown", "Unknown", 16033, "Unknown"),
]

by_designation = defaultdict(int)
for _, _, count, desig in materials:
    by_designation[desig] += count

total_system = sum(by_designation.values())
lead_lines = by_designation["Lead"]
galv_lines = by_designation["Galvanized"]
nonlead_lines = by_designation["Non-Lead"]
unknown_lines = by_designation["Unknown"]
known_hazardous = lead_lines + galv_lines

# Proportion of known lines that are hazardous → apply to unknowns
known_total = total_system - unknown_lines
hazardous_rate = known_hazardous / known_total
est_hazardous_unknowns = int(unknown_lines * hazardous_rate)
total_replacements = known_hazardous + est_hazardous_unknowns

print("=" * 60)
print("1. PIPE INVENTORY")
print("=" * 60)
print(f"   Lead lines:          {lead_lines:>7,}")
print(f"   Galvanized lines:    {galv_lines:>7,}")
print(f"   Non-Lead (safe):     {nonlead_lines:>7,}")
print(f"   Unknown:             {unknown_lines:>7,}")
print(f"   TOTAL system:        {total_system:>7,}")
print(f"\n   Known hazardous:     {known_hazardous:>7,}")
print(f"   Hazardous rate (known): {hazardous_rate:.1%}")
print(f"   Est. unknown hazardous: {est_hazardous_unknowns:>7,}")
print(f"   *** TOTAL TO REPLACE:   {total_replacements:>7,} ***")

# ============================================================
# 2. LEAP PARTICIPATION FUNNEL — How likely are people to enroll?
# ============================================================

print("\n" + "=" * 60)
print("2. LEAP PARTICIPATION FUNNEL")
print("=" * 60)

interested = 127
apps_sent = 90
applied = 82
dropped = 18
rejected = 2
completed = 42
in_progress = 20
proactive = 31
leak_loans = 59

proceeded = applied - dropped - rejected  # 62

# Key conversion rates
r_interest_to_apply = applied / interested          # 64.6%
r_apply_to_proceed = proceeded / applied            # 75.6%
r_proceed_to_complete = completed / proceeded       # 67.7%
r_overall = completed / interested                  # 33.1%

print(f"   Interested:        {interested}")
print(f"   Applied:           {applied}")
print(f"   Proceeded:         {proceeded}")
print(f"   Completed loans:   {completed}")
print(f"   In-progress:       {in_progress}")
print(f"\n   Interest → Apply:  {r_interest_to_apply:.1%}")
print(f"   Apply → Proceed:   {r_apply_to_proceed:.1%}")
print(f"   Proceed → Complete:{r_proceed_to_complete:.1%}")
print(f"   Overall conversion:{r_overall:.1%}")

# Line types from participation
print(f"\n   By line type (private side):")
print(f"     Galvanized (private): 90")
print(f"     Lead (public):        26")
print(f"     Copper (public):      45")
print(f"     Galvanized (public):   6")
print(f"     Unknown (public):     12")
print(f"     Plastic (public):      1")

# Proactive vs leak-driven
proactive_pct = proactive / (proactive + leak_loans)
print(f"\n   Proactive loans:    {proactive} ({proactive_pct:.0%})")
print(f"   Leak-driven loans:  {leak_loans} ({1-proactive_pct:.0%})")

# ============================================================
# 3. REPLACEMENT COST — From actual LEAP quotes
# ============================================================

print("\n" + "=" * 60)
print("3. REPLACEMENT COST (LEAP Quotes)")
print("=" * 60)

# Chosen contractor quotes from the real data
chosen_quotes = [
    17410, 12610, 6250, 8510, 8840, 7000, 9500, 6900, 5300, 6800,
    6250, 8200, 10000, 9000, 4800, 7500, 7600, 8000, 9600, 9200,
    6445, 6500, 5150, 9100, 9600, 6800, 6200, 9095, 9685, 6500,
    7500, 7500, 5200, 6950, 6750, 10185, 6400, 6800, 6800, 6800,
    6800, 5200, 6800, 6800, 5327.84, 6100, 7988.4, 7121, 4250,
    5488, 9070, 7200, 5800, 6875, 8550, 7650, 10063.93, 10536.74,
    9955, 5500, 8283.8, 6500, 6660, 5500,
]

# LEAP covers up to $10,000
loan_cap = 10000
loan_amounts = np.array([min(q, loan_cap) for q in chosen_quotes])

print(f"   Quotes analyzed:     {len(chosen_quotes)}")
print(f"   Mean chosen quote:   ${np.mean(chosen_quotes):,.0f}")
print(f"   Median chosen quote: ${np.median(chosen_quotes):,.0f}")
print(f"   Std dev:             ${np.std(chosen_quotes):,.0f}")
print(f"   Range:               ${min(chosen_quotes):,.0f} – ${max(chosen_quotes):,.0f}")
print(f"\n   After $10K LEAP cap:")
print(f"   Mean loan amount:    ${loan_amounts.mean():,.0f}")
print(f"   Median loan amount:  ${np.median(loan_amounts):,.0f}")
print(f"   Total (64 loans):    ${loan_amounts.sum():,.0f}")

avg_loan = loan_amounts.mean()
std_loan = loan_amounts.std()

# ============================================================
# 4. INSURANCE TRENDS — Demand signal
# ============================================================

print("\n" + "=" * 60)
print("4. INSURANCE WSL TRENDS (demand leading indicator)")
print("=" * 60)

ins_dates = ["Oct-25", "Nov-25", "Jan-26", "Feb-26", "Mar-26"]
wsl_policies = [830, 1468, 2284, 2701, 2862]
wsl_repairs = [4, 10, 20, 28, 41]
wsl_spend = [10200, 23880, 68040, 101556, 161508]
active_customers = [869, 1538, 2389, 2833, 3000]

avg_repair_cost = wsl_spend[-1] / wsl_repairs[-1]
monthly_growth = (wsl_policies[-1] / wsl_policies[0]) ** (1/5) - 1

print(f"   WSL policies:  {wsl_policies[0]} → {wsl_policies[-1]} ({wsl_policies[-1]/wsl_policies[0]:.1f}x in 5 months)")
print(f"   WSL repairs:   {wsl_repairs[0]} → {wsl_repairs[-1]}")
print(f"   Avg repair $:  ${avg_repair_cost:,.0f}")
print(f"   Active customers: {active_customers[-1]:,}")
print(f"   Monthly growth rate: ~{monthly_growth:.0%}")

# ============================================================
# 5. MONTE CARLO SIMULATION — LEAP Funding Model
# ============================================================

print("\n" + "=" * 60)
print("5. MONTE CARLO SIMULATION")
print("=" * 60)

YEARS = list(range(2026, 2038))
N_YEARS = len(YEARS)
N_SIMS = 10000

# --- Assumptions ---
# Annual home sale rate: Columbus ~5-7%, use 6% base
BASE_SALE_RATE = 0.06
# Equity extraction (refi/HELOC): ~3-4% per year nationally
BASE_EQUITY_RATE = 0.035
# Inheritance (no repayment): ~1.5% per year
BASE_INHERIT_RATE = 0.015

print(f"\n   Assumptions:")
print(f"     Total lines to replace:  {total_replacements:,}")
print(f"     Average LEAP loan:       ${avg_loan:,.0f}")
print(f"     Annual sale rate:        {BASE_SALE_RATE:.1%}")
print(f"     Annual equity extraction:{BASE_EQUITY_RATE:.1%}")
print(f"     Annual inheritance rate: {BASE_INHERIT_RATE:.1%}")
print(f"     Repayment rate/yr:       {BASE_SALE_RATE + BASE_EQUITY_RATE:.1%} (sale + equity)")
print(f"     Simulations:             {N_SIMS:,}")


def build_adoption_curve(total, base_rate, accel):
    """
    S-curve adoption with deadline pressure.
    Early: low voluntary uptake.
    Middle: growing awareness + insurance signals.
    Late: mandate enforcement spike.
    """
    schedule = []
    cumulative = 0
    for i in range(N_YEARS):
        remaining = total - cumulative
        years_left = N_YEARS - i

        if years_left <= 1:
            new = remaining
        else:
            # Urgency factor grows quadratically
            urgency = 1 + accel * (i / N_YEARS) ** 2 * 10
            rate = base_rate * urgency
            # Mandate pressure in last 3 years
            if years_left <= 3:
                rate += (1 - rate) * (0.3 * (4 - years_left) / 3)
            new = min(remaining, int(remaining * rate))

        schedule.append(new)
        cumulative += new
    return schedule


# --- Run simulations ---
print(f"\n   Running {N_SIMS:,} simulations...")

disbursements = np.zeros((N_SIMS, N_YEARS))
repayments = np.zeros((N_SIMS, N_YEARS))
outstanding = np.zeros((N_SIMS, N_YEARS))

for s in range(N_SIMS):
    # Randomize parameters
    n_lines = max(known_hazardous, int(np.random.normal(total_replacements, total_replacements * 0.05)))
    loan_avg = np.random.normal(avg_loan, std_loan / np.sqrt(len(loan_amounts)))  # std error of mean
    sale_r = np.clip(np.random.normal(BASE_SALE_RATE, 0.01), 0.02, 0.12)
    equity_r = np.clip(np.random.normal(BASE_EQUITY_RATE, 0.01), 0.01, 0.08)
    base_r = np.clip(np.random.normal(0.05, 0.01), 0.02, 0.10)
    acc = np.clip(np.random.normal(0.25, 0.05), 0.10, 0.40)

    repay_rate = sale_r + equity_r  # annual probability a loan gets repaid

    schedule = build_adoption_curve(n_lines, base_r, acc)

    # Cohort-based tracking: cohort_bal[j] = $ outstanding from year j's loans
    cohort_bal = np.zeros(N_YEARS)

    for yr in range(N_YEARS):
        # Disburse new loans
        new_loans = schedule[yr] * loan_avg
        disbursements[s, yr] = new_loans
        cohort_bal[yr] += new_loans

        # Repayments from all existing cohorts
        repaid = 0
        for c in range(yr + 1):
            r = cohort_bal[c] * repay_rate
            repaid += r
            cohort_bal[c] -= r
        repayments[s, yr] = repaid

    outstanding[s] = np.cumsum(disbursements[s] - repayments[s])


# ============================================================
# 6. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("6. RESULTS")
print("=" * 60)

total_disb = disbursements.sum(axis=1)
total_repay = repayments.sum(axis=1)
peak_outstanding = outstanding.max(axis=1)

# Loan model: city needs enough to cover peak outstanding balance
loan_mean = peak_outstanding.mean()
loan_median = np.median(peak_outstanding)
loan_p5 = np.percentile(peak_outstanding, 5)
loan_p95 = np.percentile(peak_outstanding, 95)

# Grant model: total disbursements (never repaid)
grant_mean = total_disb.mean()
grant_median = np.median(total_disb)
grant_p5 = np.percentile(total_disb, 5)
grant_p95 = np.percentile(total_disb, 95)

print(f"\n   --- LOAN-BASED MODEL ---")
print(f"   Total disbursed (mean):     ${total_disb.mean():>14,.0f}")
print(f"   Total repaid by 2037 (mean):${total_repay.mean():>14,.0f}")
print(f"   Peak outstanding (mean):    ${loan_mean:>14,.0f}")
print(f"   Peak outstanding (median):  ${loan_median:>14,.0f}")
print(f"   95% CI: [${loan_p5:,.0f}  –  ${loan_p95:,.0f}]")
print(f"   >>> 95% FUNDING LEVEL:      ${loan_p95:>14,.0f} <<<")

print(f"\n   --- GRANT-BASED MODEL ---")
print(f"   Total cost (mean):          ${grant_mean:>14,.0f}")
print(f"   Total cost (median):        ${grant_median:>14,.0f}")
print(f"   95% CI: [${grant_p5:,.0f}  –  ${grant_p95:,.0f}]")
print(f"   >>> 95% FUNDING LEVEL:      ${grant_p95:>14,.0f} <<<")

savings = total_disb - peak_outstanding
print(f"\n   --- LOAN vs GRANT ---")
print(f"   Mean savings (loan model):  ${savings.mean():>14,.0f}")
print(f"   Savings as % of grant:      {savings.mean()/grant_mean:.1%}")

# Year-by-year table
print(f"\n   --- YEAR-BY-YEAR (mean across {N_SIMS:,} sims) ---")
print(f"   {'Year':<6} {'New Lines':>10} {'Disbursed':>14} {'Repaid':>12} {'Outstanding':>14}")
print(f"   {'-'*58}")
baseline_schedule = build_adoption_curve(total_replacements, 0.05, 0.25)
for yr in range(N_YEARS):
    print(f"   {YEARS[yr]:<6} {baseline_schedule[yr]:>10,} "
          f"${disbursements[:, yr].mean():>12,.0f} "
          f"${repayments[:, yr].mean():>10,.0f} "
          f"${outstanding[:, yr].mean():>12,.0f}")

total_baseline = sum(baseline_schedule)
print(f"   {'TOTAL':<6} {total_baseline:>10,}")


# ============================================================
# 7. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 60)
print("7. GENERATING PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("LEAP Funding Model — Monte Carlo Analysis", fontsize=16, fontweight='bold', y=1.01)

# --- 1. Pipe inventory pie ---
ax = axes[0, 0]
labels = ["Lead\n(must replace)", "Galvanized\n(must replace)", "Non-Lead\n(safe)", "Unknown"]
sizes = [lead_lines, galv_lines, nonlead_lines, unknown_lines]
colors = ['#e74c3c', '#f39c12', '#27ae60', '#95a5a6']
explode = (0.05, 0.05, 0, 0)
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
ax.set_title(f"System Pipe Inventory ({total_system:,} lines)")

# --- 2. Participation funnel ---
ax = axes[0, 1]
stages = ['Interested', 'Applied', 'Proceeded', 'Completed']
values = [interested, applied, proceeded, completed]
colors_f = ['#3498db', '#2980b9', '#2471a3', '#1a5276']
bars = ax.barh(stages[::-1], values[::-1], color=colors_f[::-1])
for bar, val in zip(bars, values[::-1]):
    pct = val / interested * 100
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{val} ({pct:.0f}%)', va='center', fontsize=10)
ax.set_xlabel('Homeowners')
ax.set_title('LEAP Participation Funnel')

# --- 3. Quote / loan distribution ---
ax = axes[0, 2]
ax.hist(chosen_quotes, bins=15, color='#e67e22', edgecolor='black', alpha=0.8, label='Chosen quotes')
ax.axvline(x=loan_cap, color='red', linestyle='--', lw=2, label=f'$10K LEAP cap')
ax.axvline(x=np.mean(chosen_quotes), color='blue', linestyle='--', lw=2,
           label=f'Mean: ${np.mean(chosen_quotes):,.0f}')
ax.set_xlabel('Cost ($)')
ax.set_ylabel('Count')
ax.set_title('Contractor Quote Distribution')
ax.legend(fontsize=9)

# --- 4. Outstanding fund over time ---
ax = axes[1, 0]
mean_out = outstanding.mean(axis=0)
p5_out = np.percentile(outstanding, 5, axis=0)
p95_out = np.percentile(outstanding, 95, axis=0)
ax.fill_between(YEARS, p5_out / 1e6, p95_out / 1e6, alpha=0.25, color='#3498db')
ax.plot(YEARS, mean_out / 1e6, 'b-o', lw=2, label='Mean')
ax.plot(YEARS, p95_out / 1e6, 'b--', lw=1, alpha=0.6, label='95th %ile')
ax.set_xlabel('Year')
ax.set_ylabel('$ Millions')
ax.set_title('Loan Model: Net Outstanding Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# --- 5. Loan vs Grant bar chart ---
ax = axes[1, 1]
x = ['Loan Model\n(peak outstanding)', 'Grant Model\n(total cost)']
heights = [loan_p95 / 1e6, grant_p95 / 1e6]
bar_colors = ['#27ae60', '#e74c3c']
bars = ax.bar(x, heights, color=bar_colors, edgecolor='black', width=0.5)
for bar, h in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'${h:.0f}M', ha='center', fontweight='bold', fontsize=13)
ax.set_ylabel('$ Millions')
ax.set_title('95% Confidence Funding Level')
ax.grid(True, axis='y', alpha=0.3)

# --- 6. Funding distribution histograms ---
ax = axes[1, 2]
ax.hist(peak_outstanding / 1e6, bins=50, color='#27ae60', alpha=0.7,
        edgecolor='black', label='Loan (peak)')
ax.hist(total_disb / 1e6, bins=50, color='#e74c3c', alpha=0.5,
        edgecolor='black', label='Grant (total)')
ax.axvline(loan_p95 / 1e6, color='green', ls='--', lw=2, label=f'Loan 95%: ${loan_p95/1e6:.0f}M')
ax.axvline(grant_p95 / 1e6, color='red', ls='--', lw=2, label=f'Grant 95%: ${grant_p95/1e6:.0f}M')
ax.set_xlabel('Funding Required ($M)')
ax.set_ylabel('Frequency')
ax.set_title('Simulation Distribution')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/leap_funding.png", dpi=150, bbox_inches='tight')
print(f"   Saved: {OUT}/leap_funding.png")

# --- Supplementary: Insurance demand signal ---
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Insurance Program — LEAP Demand Leading Indicator", fontsize=14, fontweight='bold')

ax1.plot(ins_dates, wsl_policies, 'g-s', lw=2, ms=8, label='WSL Policies')
ax1.plot(ins_dates, active_customers, 'b-o', lw=2, ms=8, label='Active Customers')
ax1.set_ylabel('Count')
ax1.set_title('Policy & Customer Growth')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(ins_dates, [s/1000 for s in wsl_spend], color='#e74c3c', edgecolor='black')
ax2_twin = ax2.twinx()
ax2_twin.plot(ins_dates, wsl_repairs, 'b-o', lw=2, ms=8)
ax2_twin.set_ylabel('Repairs', color='blue')
ax2.set_ylabel('Repair Spend ($K)')
ax2.set_title('WSL Repair Volume & Spend')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/leap_insurance_signal.png", dpi=150, bbox_inches='tight')
print(f"   Saved: {OUT}/leap_insurance_signal.png")

# --- Supplementary: Adoption curve ---
fig3, ax = plt.subplots(figsize=(10, 5))
sched = build_adoption_curve(total_replacements, 0.05, 0.25)
cum = np.cumsum(sched)
ax.bar(YEARS, sched, color='#3498db', edgecolor='black', alpha=0.8, label='Replacements/year')
ax2 = ax.twinx()
ax2.plot(YEARS, cum, 'r-o', lw=2, ms=6, label='Cumulative')
ax2.axhline(y=total_replacements, color='red', ls='--', alpha=0.5, label=f'Target: {total_replacements:,}')
ax.set_xlabel('Year')
ax.set_ylabel('Lines Replaced')
ax2.set_ylabel('Cumulative Replacements')
ax.set_title('Projected Replacement Timeline (S-curve adoption + mandate pressure)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/leap_adoption_curve.png", dpi=150, bbox_inches='tight')
print(f"   Saved: {OUT}/leap_adoption_curve.png")

print("\n" + "=" * 60)
print("EXECUTIVE SUMMARY")
print("=" * 60)
print(f"""
City of Columbus LEAP Funding Model
────────────────────────────────────
PIPE INVENTORY
  {lead_lines:,} lead + {galv_lines:,} galvanized = {known_hazardous:,} known
  + ~{est_hazardous_unknowns:,} est. from {unknown_lines:,} unknowns
  = {total_replacements:,} total lines to replace by 2037

COSTS
  Avg contractor quote: ${np.mean(chosen_quotes):,.0f}
  Avg LEAP loan (≤$10K): ${avg_loan:,.0f}

ADOPTION
  Current conversion: {r_overall:.0%} of interested → completed
  65% of loans are leak-driven (reactive, not proactive)
  Insurance WSL: 3.4x growth in 5 months → demand accelerating

FUNDING (95% confidence, {N_SIMS:,} simulations)
  ┌─────────────┬────────────────┐
  │ Loan model  │ ${loan_p95/1e6:>10.1f}M │
  │ Grant model │ ${grant_p95/1e6:>10.1f}M │
  │ Savings     │ ${(grant_p95-loan_p95)/1e6:>10.1f}M │
  └─────────────┴────────────────┘
  Loan model saves ~{savings.mean()/grant_mean:.0%} vs grants

  95% CI (loan):  ${loan_p5/1e6:.0f}M – ${loan_p95/1e6:.0f}M
  95% CI (grant): ${grant_p5/1e6:.0f}M – ${grant_p95/1e6:.0f}M
""")
