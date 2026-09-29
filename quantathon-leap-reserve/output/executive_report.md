# Executive Summary: Columbus LEAP Funding Need

## Bottom Line

Reserve **$2.73M in 2026** under the loan-based LEAP structure. This amount keeps the fund solvent through 2037 in **95% of 3,000 simulated futures** under base-case assumptions. Model validation: **PASS**.

If the City funds the program as a pure grant with no repayments, the equivalent 95%-safe reserve rises to **$4.01M**. The deferred-repayment loan structure reduces required upfront capital by **$1.27M**.

## Key Numbers

| Metric | Loan Model | Grant Model |
|---|---:|---:|
| Expected funding need | $2.49M | $3.71M |
| **95%-safe funding (P95)** | **$2.73M** | **$4.01M** |
| 95% uncertainty range | $2.20M – $2.77M | $3.36M – $4.06M |
| Capital savings (loan vs grant) | **$1.27M** | — |

## Scenario Comparison

| Scenario | Loan P95 | Grant P95 |
|---|---:|---:|
| Optimistic | $2.01M | $3.16M |
| **Base** | **$2.73M** | **$4.01M** |
| Conservative | $3.49M | $4.77M |

## Stress Test Results

| Scenario | P95 Funding | vs Base |
|---|---:|---:|
| 2029 Housing Crash | $3.14M | +14.9% |
| Surge Uptake | $4.98M | +82.1% |
| Perfect Storm | $5.75M | +110.7% |

Even under the worst-case "Perfect Storm" (housing crash + surge uptake combined), the P95 funding requirement remains below $6M — manageable for a city-scale program.

## What Drives the Answer

The biggest risk driver is **LEAP uptake rate** — it accounts for **78% of parameter variance**. If more households seek LEAP before their scheduled replacement, required funding rises quickly. Repayment mechanics (sale probability, equity extraction) are secondary drivers at ~4% each.

## Early-Warning Monitoring Triggers

Track these three KPIs against the model's expectations:

| KPI | Median | Warning Threshold (P75/P25) | Action if Breached |
|---|---:|---:|---|
| Cumulative originations by 2028 | 183 | > 193 | Review reserve adequacy |
| Average funded amount | $8,252 | > $8,312 | Increase inflation assumption |
| Cumulative repayments by 2029 | 27 | < 23 | Review repayment assumptions |

## Model Confidence

- Deterministic-MC gap: **-0.72%** (two independent models agree)
- Bootstrap CI on P95: [$2.72M, $2.74M]
- Coefficient of variation: **0.0011** (high precision)
- NPV sensitivity to discount rate: only +/-3% across the 1%-5% range

## Recommendation

Set the **2026 loan-based LEAP reserve at $2.73M**. If decision-makers prefer extra cushion, the conservative scenario suggests **$3.49M**. For maximum resilience against combined adverse events, plan for **$5.75M**.

The single most valuable future data investment is a multi-year LEAP uptake history — resolving uptake uncertainty would reduce the model's confidence interval by ~78%.
