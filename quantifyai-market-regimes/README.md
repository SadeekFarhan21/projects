# quantifyai-market-regimes

Audit material for the blog post about **QuantifyAI** (the repo's own README calls it "Quantathon"), a team submission to the 2025 Ohio State Quantathon. Upstream repository: https://github.com/SadeekFarhan21/QuantifyAI (public, no license file). The team's source code is not copied here. This folder holds only what I wrote after the fact.

## What is here

- `tools/ablate.py`: the post-hoc audit script. It was written after the fact, not by the team, and needs the upstream `src/` on the path.
- `results/`: outputs of my reproduction run on the upstream code at commit ed236e7 (2025-03-02): `enhanced_performance_metrics.csv`, `pipeline_run.log`, `ablation_output.txt`, `markov_run_failure.log`, `confusion_matrix.png` (the figure committed upstream).

## Credit

From the upstream git history (32 commits): Farhan Sadeek 17, Jalen Francis 9, Aditya Bhati 2, Andrew McKenzie 1. The paper and slides also list Jayson Clark as an author. Jalen Francis committed the modular codebase in one commit, so git cannot show who wrote which part before that merge.

## Reproducing

Clone upstream, add the Quantathon workbook (competition data, not included) at `data/market_data.xlsx`, use pandas below 3, then:

```
python main.py --data data/market_data.xlsx --output out --start_date 2019-01-01 --end_date 2022-12-31 --train_end_date 2018-12-31 --enhanced
python /path/to/quantifyai-market-regimes/tools/ablate.py out
```

Environment used: Python 3.14.7, pandas 2.3.3, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2.
