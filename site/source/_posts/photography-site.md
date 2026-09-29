---
layout: post
title: "Timestamps and Hashes Did Most of My Photo Culling"
code: https://github.com/SadeekFarhan21/projects/tree/main/photography-site
date: 2024-11-18 11:33:22
tags:
  - photography
  - computer-vision
  - nextjs
description: >-
  A curation pipeline for my photo portfolio scored 1,803 images and kept 217,
  and most of the cuts came from grouping near-duplicates by capture time and
  perceptual hash, not from the vision model's aesthetic score.
---

I shoot city and landscape travel photos, and my Google Photos library had **2,952 images** in it, thousands of them near-identical frames: burst shots, drone passes and repeat attempts at the same skyline. I built a pipeline to turn that pile into a portfolio. Classical filters catch blur and bursts, a vision model scores each photo against a written rubric, a second pass ranks the survivors and writes titles and captions, and a bulk uploader pushes the result into a Next.js photo site.

The main lesson is that the cheap signals did most of the work. The cloud scoring run scored **1,803 photos and kept 217** at a threshold of 70, about 12 percent, and **most of the cuts came from grouping near-duplicates by capture time and perceptual hash, not from the aesthetic score**. The small local model I tried first did even less: a 3B vision model on a laptop put 1,669 of 2,419 photos in the 20s and removed only 13.

## Why It Matters

Every photographer knows this problem. The one good frame sits next to a dozen near-identical ones, and picking by hand from about 2,950 files is slow, boring work. I wanted a machine to make the first cut.

There are two sites. The first was the quick answer: a Jekyll gallery, [PhotographyWebsite](https://github.com/SadeekFarhan21/PhotographyWebsite), built on rampatra's photography template<sup>[[1]](#ref-1)</sup> and AJ's html5up design<sup>[[2]](#ref-2)</sup>, with 52 photos I picked by hand, deployed on Vercel. The second was the ambitious one: a database-backed gallery with EXIF, location pages and proper storage, fed by a curated set instead of whatever I remembered to upload. It is a fork of Sam Becker's exif-photo-blog<sup>[[3]](#ref-3)</sup>, and the curation pipeline sits next to it.

I kept the pipeline's job narrow. Cut the library down mechanically, rank what's left with a vision model against a rubric, generate titles, captions and tags, and upload with the EXIF data and blur placeholders the site expects.

## Technical Details

### Two Sites, No Shared Code

<figure class="excal" data-diagram="photography-site-architecture"><a href="/img/diagrams/photography-site-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/photography-site-architecture.webp" alt="Two lanes: the public Jekyll repo takes 52 curated photos through a gulp ImageMagick resize, a Jekyll build driven by _config.yml and Vercel static hosting, crediting rampatra's template and AJ's design, while the private photo-graphy repo takes Google Photos (2,952 files) through blur and pHash filters, gpt-4o-mini per-month scoring, shooting-session clustering that keeps the best frame and a second brutal ranking pass, then upload_photos.mjs writes to Postgres and object storage that a Next.js exif-photo-blog fork reads, with the local Ollama attempt off this path because its 3B pass cut only 13 of 2,419." width="2400" height="1422" loading="lazy" decoding="async"></a></figure>

The public path is linear. Photos go through a gulp resize task, Jekyll builds the static pages, and Vercel serves them.

The curation path has six stages. Google Photos access produces a folder of files organised by month. Classical filters remove blurry frames and bursts. A cloud model scores each month in batches. Shooting-session clustering collapses what's left, keeping the best frame per cluster. A second, harsher ranking pass orders the survivors, and metadata is generated for them. Finally an uploader writes the originals and resized variants to object storage and a row per photo into Postgres, where the exif-photo-blog fork reads them.

### Cheap Filters First

Before any model sees a photo, two classical checks are nearly free. Blur can be estimated as the variance of the Laplacian of the image: a sharp image has strong edges and a high variance<sup>[[4]](#ref-4)</sup>. Duplicates and bursts can be found with perceptual hashes, which map visually similar images to hashes that differ in only a few bits. A difference hash (dHash) compares the brightness of adjacent pixels in a tiny grayscale copy<sup>[[5]](#ref-5)</sup>, and the DCT-based pHash is its heavier relative<sup>[[6]](#ref-6)</sup>. The distance between two hashes is the Hamming distance, the number of bits that differ. `curate.py` uses pHash, and `curate_month.py` uses its own 64-bit dHash, treating a Hamming distance of 6 or less as a duplicate.

### Session Clustering by Time

Photos from one burst or one drone flight are taken seconds apart. Sort by capture time, start a new group whenever the gap exceeds a threshold, and you've grouped them without looking at a single pixel. I used a 30 second gap. Combining that with the hash distance also catches the same scene shot twice with a pause in between.

<figure class="excal" data-diagram="photography-site-clustering"><a href="/img/diagrams/photography-site-clustering.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/photography-site-clustering.webp" alt="An illustrative timeline of photo capture times split into three sessions wherever the gap exceeds 30 seconds, followed by a flow that dHashes every frame, merges at Hamming distance 6 or less, clusters near-duplicates across the month and keeps the highest score in each cluster, with settings of 30 s, Hamming <= 6 and threshold 70, the real 2025-07 effect of 330 of 488 images excluded as near-duplicates." width="2400" height="1246" loading="lazy" decoding="async"></a></figure>

### Prompted Scoring Is a Coarse Instrument

Asking a vision language model for a 0 to 100 aesthetic score is an LLM-as-judge setup<sup>[[7]](#ref-7)</sup>. It's cheap and it can follow a written rubric, but the scores aren't calibrated. A small model may squeeze everything into a narrow band, which is exactly what I saw, and even a strong one is a filter, not an oracle. That shaped the design: use the model to throw out the clearly weak, not to crown the best.

## Implementation

### The Public Site Is Mostly Configuration

The Jekyll site is rampatra's open-source photography template<sup>[[1]](#ref-1)</sup>, which builds on a design by AJ at html5up<sup>[[2]](#ref-2)</sup>. My part is the photos, some resizing and `_config.yml`, which also sets which EXIF tags the click-through popup shows.

```yaml
title: "Farhan's Photography"
exif: '[{"tag": "Model", "icon": "camera-retro"}, {"tag": "FNumber", "icon": "dot-circle-o"}, {"tag": "ExposureTime", "icon": "clock-o"}, {"tag": "ISOSpeedRatings", "icon": "info-circle"}]'
```

The template ships a gulp task that resizes originals into full and thumbnail sizes with ImageMagick.

```js
gulp.task('resize-images', function () {
    return gulp.src('images/*.*')
        .pipe(imageResize({ width: 1024, imageMagick: true }))
        .pipe(gulp.dest('images/fulls'))
        .pipe(imageResize({ width: 512, imageMagick: true }))
        .pipe(gulp.dest('images/thumbs'));
});
```

The site holds 52 full-size images (32,588 KB) and 52 thumbnails (6,964 KB). Vercel Speed Insights sits in the footer.

### The Curation Pipeline

The curation scripts live in a private repository, so I describe them here rather than quote them.

`curate.py` is the first pass. It computes the variance of the Laplacian for each image, clusters bursts by perceptual hash and keeps the sharpest of each, then writes contact sheets and a `report.csv`. `score_photos.py` sends each image, downscaled to a 768 px JPEG, to an OpenAI-compatible endpoint at temperature 0.2 and expects JSON back with a `score` from 0 to 100 and a `reason`. It appends to `scores.csv` as it goes, so a killed run picks up where it stopped.

My first plan, `run_pipeline.sh`, ran two local passes with Ollama: qwen2.5vl:3b<sup>[[8]](#ref-8)</sup> over everything with a minimum score of 15, then the 7B model over the survivors, keeping the top 300. `curate_month.py` replaced it. It scores one month at a time with gpt-4o-mini<sup>[[9]](#ref-9)</sup>, 10 images per request across 5 workers, keeps images scoring 70 or more, and clusters shooting sessions with the 30 second gap and the Hamming distance of 6, keeping the best-scoring frame in each cluster.

### The Rubric

The prompt asks for a harsh scale for a city and landscape travel portfolio. People can appear in scenes, but portraits and selfies score low. Glare costs 10 points. When several images are shown together, near-duplicates cost 20 points, which I called the variety rule. Sharpness is deliberately left out of the rubric, because the model only ever sees a downscaled copy. The second pass, `prompt_pass2.txt`, is brutal on purpose: most photos should land in the 50s and 60s, and only a handful should reach 90.

### Metadata and Upload

`generate_metadata.py` works through the photos in the order the second pass ranked them. It reads EXIF from the local files and asks gpt-4o-mini for a title, caption, tags and location.

`upload_photos.mjs` re-implements the exif-photo-blog upload path so it can run in bulk. For each photo it writes the original plus small, medium and large variants to storage, reads EXIF with exifr, computes a blur placeholder with sharp, generates a nanoid and inserts the row. Every upload is logged to `_uploaded.json`, so a run can be rolled back.

### My Changes to the exif-photo-blog Fork

The fork is Sam Becker's work<sup>[[3]](#ref-3)</sup>, with a long upstream history. On top of it I added:

- **Location collections**, with `/location/[location]` pages and an admin CRUD for locations. That change touched 55 files (+1,484/−35) and also added platform files such as `dji.ts`, `samsung.ts` and `sony.ts`.
- **Camera label normalisation**, category sorting by count, and a three-column photo grid.
- **A migration from Vercel Blob to Cloudflare R2**, along with `generate-r2-ai-metadata.mjs` and `sync-r2-exif-metadata.mjs`. The migration script runs dry by default. With `--execute` it copies every object, including the optimised variants, under the same key, rewrites the URL in Postgres, and can resume after an interruption.

## Problems

### 1. The Local 3B Model Squeezed Every Score Into One Band

The two-pass local design assumed a small model could throw out the obvious junk cheaply. It couldn't. **Of 2,419 scored photos, 1,669 landed in the 20s, and the cutoff of 15 removed just 13.**

<figure data-figure="chart:projects/photography-site/photography-site-score-histogram"></figure>

There's a second hump in the 70s, with 289 photos, so the model wasn't scoring at random. But **a cutoff that catches only the bottom 0.5 percent leaves the 7B pass with almost everything**, which defeats the point of a cheap first pass. A filter that removes 13 of 2,419 isn't a filter. So I dropped the local plan and moved to per-month cloud scoring with gpt-4o-mini, with deduplication doing the heavy lifting before the threshold.

### 2. Batching Changes What a Score Means

Scoring 10 images per request lets the model compare them, and the variety rule depends on that. It also means **a photo's score depends on what it was shown with**. I chose batching anyway, for cost and for the duplicate penalty. I didn't measure how far a photo's score moves between batches, so I know the effect exists and nothing about its size.

## Results

### The Local Pass Barely Filtered

The histogram above is the clearest result. The 3B model scored all 2,419 photos with no errors. Only 13 fell below the cutoff of 15, and 2,406 passed. The mass sits in the 20s (1,669), with smaller groups in the 30s (143), 40s (136), 60s (117), 70s (289), 80s (32) and 90s (20). **A small local model on a laptop compressed nearly everything into one band.**

### The Cloud Run Kept About 12 Percent

<figure data-figure="chart:projects/photography-site/photography-site-curation-funnel"></figure>

The library manifest lists 2,952 images, 2,750 camera originals and 202 other files. Across the months in the logged run, the cloud pipeline **scored 1,803 images and kept 217 at threshold 70, with 0 scoring errors**. The final upload run pushed 157 of the 160 photos it planned to the site, and `_uploaded.json` holds 180 rows once earlier uploads are counted.

### Time and Hashes Did More Than Aesthetics

Clustering had a bigger effect than scoring. In the largest logged month, 2025-07, **330 of 488 images were excluded as near-duplicates** before the threshold decided anything. That's 68 percent of the month removed by capture time and perceptual hash alone. **It's a cheaper and more reliable signal than asking a model whether a photo is beautiful.**

## What I Would Change

### Hold Out a Labelled Sample

I have no measure of whether the 217 are the right 217, and I haven't evaluated the second pass's ranking either. The most valuable addition would be 100 photos I label by hand as keep or drop. Then every change to the prompt, model or threshold gets a number, agreement and precision on the kept set, instead of a judgement call.

### Measure Score Stability

Score the same month twice and, separately, shuffle the batches, then report how much scores move. That would show whether the threshold of 70 sits in a noisy region and whether batching helps or hurts.

## References

1. <span id="ref-1"></span>rampatra. *photography: A Jekyll website for photographers*. GitHub. [link](https://github.com/rampatra/photography)
2. <span id="ref-2"></span>AJ. *Multiverse*. HTML5 UP. [link](https://html5up.net/multiverse)
3. <span id="ref-3"></span>Sam Becker. *exif-photo-blog: a Next.js photo blog with EXIF data*. GitHub. [link](https://github.com/sambecker/exif-photo-blog)
4. <span id="ref-4"></span>José Luis Pech-Pacheco, Gabriel Cristóbal, Jesús Chamorro-Martínez, Joaquín Fernández-Valdivia. *Diatom autofocusing in brightfield microscopy: a comparative study*. International Conference on Pattern Recognition (ICPR), 2000. [link](https://doi.org/10.1109/ICPR.2000.903548)
5. <span id="ref-5"></span>Neal Krawetz. *Kind of Like That*. The Hacker Factor Blog, 2013. [link](https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
6. <span id="ref-6"></span>Christoph Zauner. *Implementation and Benchmarking of Perceptual Image Hash Functions*. Master's thesis, Upper Austria University of Applied Sciences, 2010. [link](https://www.phash.org/docs/pubs/thesis_zauner.pdf)
7. <span id="ref-7"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS Datasets and Benchmarks, 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
8. <span id="ref-8"></span>Shuai Bai, Keqin Chen, Xuejing Liu et al. *Qwen2.5-VL Technical Report*. arXiv, 2025. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
9. <span id="ref-9"></span>OpenAI. *GPT-4o mini: advancing cost-efficient intelligence*. OpenAI blog, 2024. [link](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
