# Airport network analysis (Global Flight Connectivity)

A single Quarto/R notebook (`index.qmd`, "Computational Notebook: Global Flight Connectivity") that builds a directed airport network with igraph from a 5,065-row sample of monthly flight lists (Mar 2020 - Oct 2021) and runs descriptive network analysis in the style of Kolaczyk and Csardi's *Statistical Analysis of Network Data with R* (degree, strength, assortativity, connectivity, transitivity, centralities).

- Author: Farhan Sadeek (solo repo, 15 commits, 2026-02-05 to 2026-08-12).
- Upstream: https://github.com/SadeekFarhan21/Airport-Network-Analysis (branch master, HEAD ef657f6). Published at https://airport.farhansadeek.com.
- License: the upstream repo has no LICENSE file.
- Context: it appears to be a course assignment (SAND textbook, "Chapter 4" commit message). The early commits a3bf59b, 1fd65c9 and ec948a3 load an ego-network CSV from a `notes.farhansadeek.com/dartmouth/math7/homework/` URL, which points to Dartmouth MATH 7 homework; the README itself names no course.
- AI involvement: 9 of 15 upstream commits carry Claude co-author trailers. Eight (Sonnet 4.5, all 2026-02-05) are repo/Quarto/Netlify setup and data organisation; one (1fd65c9, Opus 4.6) is an analysis commit. `index.qmd` labels the centrality bubble plot and the basic network property checks as made with Claude Code, and earlier commits have a `## Gemini 3.1 Pro` section. Attribution of the remaining code is not recorded; the analysis prose is Farhan's by his account.

## What is and is not in this folder

Copied: `index.qmd`, Quarto/Netlify build config (`_quarto.yml`, `netlify.toml`, `build.sh`, `publish.sh`), `styles.css`, the RStudio project file, and the upstream README as `UPSTREAM_README.md`.

Not copied: the 20 raw monthly flight lists (about 8 GB, Git LFS), `sampled_flights.csv` (the notebook input; provider and license not stated upstream, so it is not redistributed), generated build output, the Netlify site id file, and unrelated files. The code that produced `sampled_flights.csv` is not in the upstream repo, so its sampling method is unknown.

Data provenance is not stated upstream. Filenames and columns resemble the OpenSky Network COVID-19 flight lists; that is an unverified inference.

## Reproduce

Needs R with tidyverse, igraph, sand, igraphdata, ggraph, packcircles, plus Quarto. Put the sample CSV beside `index.qmd` (fetch from upstream via Git LFS), then:

```
quarto render index.qmd --to html
Rscript results/run.R    # writes the numbers in results/run.out
```

`results/run.R` re-implements the notebook's graph-building pipeline (same drop_na, group_by, simplify steps) to print the statistics; `results/run.out` is its output (igraph 2.2.1). Numbers were not compared with the live site.

## Notebook claims that did not survive recomputation

See `results/run.out`: vertex/edge connectivity is 0 (graph is disconnected, not "connected to every airport"); KCLT is 6th by degree and KIAD is not in the top 10; KTPA is not in the top 10 by closeness; Spearman strength-betweenness (0.898) exceeds strength-eigenvector (0.672), the reverse of the notebook's claim.

## Added checks

`results/weights_check.R` / `.out`: igraph uses the `weight` edge attribute (flights per route) by default in betweenness, closeness and eigenvector centrality, so the notebook's tables are weighted; the script also prints unweighted values, closeness ranks (KTPA 81, KCLT 32, KIAD 13 weighted) and zero-betweenness counts. `results/eig_stability.R` / `.out`: Spearman correlations involving eigenvector centrality vary between reruns (0.599 to 0.672) because ~732 airports have near-zero scores.
