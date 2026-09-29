# ListenTing

Hackathon build (Anthropic Cultural Exchange Hackathon, per the slide title): a listening-first Chinese learning PWA for heritage speakers. It plays Sun Tzu's *Art of War* (孙子兵法) as audio with a synced transcript, idiom (chengyu) cards, a Claude-generated spoken comprehension check-in graded by Claude, and a saved-vocab vault.

Upstream: https://github.com/SadeekFarhan21/ListenTing (copied from HEAD `ae0220b`). The upstream repo has no LICENSE file. This is a partial copy for a blog write-up; see `UPSTREAM_README.md` for the original README.

## Credits (from git history)

- Farhan Sadeek: the three pipeline scripts, the Next.js app (API routes, Player, CheckinModal, idiom data, PWA/service worker), and the Astro slide deck. 11 commits plus a merge.
- Alex (alextang2763): repo organization, UTF-8 conversion of the source text, chapter images, and the tap-to-select vault UI (later reverted in part).
- Mulong (Mulong109): uploaded the source PDF, per-chapter text and the chapter 1-4 MP3s that the app ships.
- Vercel bot: Web Analytics commit.
- 12 of the 27 commits carry Claude co-author trailers (Sonnet 4.6 on 8, Opus 4.7 on 4). Blame cannot separate hand-written from AI-written code.

## What is here

- `apps/web/`: Next.js 15 / React 19 / Tailwind app, and `scripts/` (offline pipeline: `prepare-chapters.ts`, `enrich-translations.ts`, `generate-audio.ts`, hand-curated `idioms.ts`, chapter titles).
- `scripts/measure.py`: counts sentences, idiom tags, and (with ffprobe) real vs synthetic durations.
- `results/`: measurements produced for the post.

## Deliberately not copied

Third-party source text (qinkan.net Chinese/modern-Chinese translation, PDFs), `public/chapters/*.json` (embeds that text), the ElevenLabs MP3s (15 MB), AI-generated chapter images, the slide deck, lockfile, `.env*`, and `.claude/settings.json`. Without `chinese/` the pipeline cannot be re-run from this copy.

## Measured results (no paid APIs)

Commands, run in a scratch copy of the upstream repo:

```
cd apps/web && npx tsx scripts/prepare-chapters.ts        # results/prepare-chapters.log
python3 scripts/measure.py <committed chapters> <committed audio>   > results/committed_measurements.json
python3 scripts/measure.py <regenerated chapters>                   > results/regenerated_measurements.json
```

- 13 chapters, 290 sentences, 169 idiom tags (0.58 per sentence). Regeneration reproduces the committed JSON exactly except chapter 11 (a stray `~` in one gloss that was hand-fixed in the committed file).
- English translation field filled for 0 of 290 sentences: `enrich-translations.ts` was never run on the committed data. The `gloss` field (modern Chinese from the source) is empty for 20 of 290.
- Chapters 1-4 audio: 230.43 / 213.84 / 278.18 / 221.88 s, versus the committed synthetic timelines of 88.08 / 85.83 / 107.57 / 77.94 s (4.5 chars/sec estimate), i.e. 2.49x to 2.85x longer (2.63x overall). Transcript timestamps would drift against those MP3s until `generate-audio.ts` rewrites them. Not tested in the running app.

## Not measured

The full app was not run (needs `ANTHROPIC_API_KEY` and `ELEVENLABS_API_KEY`). Translation quality, grading accuracy, TTS cost and latency are unmeasured. The slide claims (~$0.01 per session, 3.5M heritage speakers, 2500 years of commentary) have no support in the repo and are not repeated here.
