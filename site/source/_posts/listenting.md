---
layout: post
title: "Learning Chinese by Ear From Sun Tzu"
code: https://github.com/SadeekFarhan21/projects/tree/main/listenting
date: 2026-05-23 09:30:30
tags:
  - team-project
  - llm
  - nextjs
  - audio
description: "A weekend hackathon PWA that plays Sun Tzu in Chinese with a synced transcript, 169 idiom tags across 290 sentences, and a spoken comprehension check-in that Claude writes and grades."
---

Heritage speakers understand spoken Chinese but can't read it well, and most learning apps start from characters. ListenTing starts from listening. It plays Sun Tzu's *Art of War*<sup>[[1]](#ref-1)</sup> in Chinese with a transcript that highlights the current sentence, flags idioms in the text, and every three minutes by default pauses to ask a spoken comprehension question that Claude writes and grades. Three of us built it over one weekend, 2026-05-23 to 05-25, for the Anthropic Cultural Exchange Hackathon.

The design idea is to do the expensive work once, offline, so it isn't paid for by every listener. A pipeline segments the source into **13 chapters, 290 sentences and 169 idiom tags**, translates it with Claude Haiku 4.5, and narrates it with ElevenLabs one sentence per call, so each sentence's place in the audio comes from its own clip. The app is a Next.js PWA that loads those files, plays the chapter, and uses Haiku to write each check-in question and Sonnet to grade the answer.

## Why It Matters

A heritage speaker can follow the sound of Chinese and stalls at the page. I wanted an app where the audio is the main thing and the text is scaffolding you can lean on.

Two ideas sit behind that. Heritage language learners are a distinct group whose listening and speaking outrun their literacy<sup>[[2]](#ref-2)</sup>. And the comprehensible-input hypothesis argues that acquisition is driven by understanding messages slightly above your current level<sup>[[3]](#ref-3)</sup>. Put together, they point to a product where you hear real Chinese at natural pace and the transcript is a fallback.

We picked Sun Tzu because it is short, ancient, and full of four-character idioms that people still use. A phrase like 知己知彼, "know yourself and know your opponent", is both a line in the book and something a heritage speaker may have heard at a dinner table. Tapping it to see the literal reading and the modern meaning connects two things they half know.

The weekend scope was a player with a highlighted transcript, idiom cards, a vault for saved words, a scheduled spoken check-in, an installable PWA, and an offline pipeline, because translating at runtime would cost money for every listener.

## Technical Details

### A Synced Transcript Is a Timeline

A synced transcript has one core data structure: a list of sentences, each with a start and end time in the audio. The player asks which sentence contains the current audio time and highlights it. If the times are wrong, the highlight is wrong, however good everything else is.

The cleanest way to get the times is to measure them from the audio, one sentence at a time. That is what the narration script does. It asks ElevenLabs for one clip per sentence, measures each clip, and writes the start and end of every sentence from those measured lengths.

### Measuring MP3 Duration Without ffmpeg

The narration script measures each generated clip by walking the MP3 frame headers. For MPEG-1 Layer III every frame holds 1,152 samples, and its length in bytes is

$$
\text{frame length} = \left\lfloor \frac{1152}{8} \cdot \frac{\text{bitrate}}{\text{sample rate}} \right\rfloor + \text{padding}
$$

so summing frames and dividing the total samples by the sample rate gives the duration. That is exact for constant-bitrate audio, and the script requests `mp3_44100_128` from ElevenLabs.<sup>[[6]](#ref-6)</sup> The frame formula is standard MP3 framing, not something from that documentation.

### Architecture

The system has an offline half and an online half, joined by static files.

<figure class="excal" data-diagram="listenting-architecture"><a href="/img/diagrams/listenting-architecture.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/listenting-architecture.webp" alt="ListenTing architecture in two halves: an offline pipeline where qinkan.net text goes through prepare-chapters.ts into public/chapters/1..13.json (13 chapters, 290 sentences, 169 idiom tags), which feeds enrich-translations.ts (Haiku 4.5 fills en) and generate-audio.ts (ElevenLabs per sentence, rewrites timestamps from real durations), plus ch1-4.mp3 ElevenLabs narration uploaded by Mulong, feeding a browser PWA (Player, CheckinModal, IdiomCard with 28 curated entries, VaultSheet on device) that calls the Next.js API routes /api/question, /api/grade, /api/speak, /api/transcribe and /api/translate." width="2400" height="1769" loading="lazy" decoding="async"></a></figure>

The offline half lives in `apps/web/scripts`. `prepare-chapters.ts` reads the source text and writes one JSON per chapter into `public/chapters`. `enrich-translations.ts` fills an English field on each sentence with Claude Haiku 4.5, and `generate-audio.ts` narrates each sentence with ElevenLabs and writes the timestamps from the real durations.

The online half is a Next.js 15 and React 19 app with Tailwind, packaged as a PWA with a small service worker. The player loads a chapter's JSON as a static file, asks the server whether `/audio/chN.mp3` exists, and plays it if it does. If not, it falls back to the browser's speech synthesis and walks the sentences one by one. The API routes hold the keys: Haiku 4.5 writes the check-in question, Sonnet 4.5 grades the answer, ElevenLabs reads the question aloud, and `/api/translate` translates on demand.

The check-in has its own small state machine, and it is where the model calls live.

<figure class="excal" data-diagram="listenting-checkin"><a href="/img/diagrams/listenting-checkin.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/listenting-checkin.webp" alt="The spoken check-in as a vertical chain of phases in CheckinModal.tsx: the player pauses after 3 minutes of audio, loading calls /api/question (Haiku 4.5, last 12 sentences), asking calls /api/speak with a speechSynthesis fallback, listening uses browser SpeechRecognition or typed input, grading calls /api/grade (Sonnet 4.5) and the result phase shows a verdict of great, good, partial or off with feedback under 30 words." width="2400" height="2640" loading="lazy" decoding="async"></a></figure>

## Implementation

I wrote the pipeline scripts, the Next.js app and the slide deck. Alex organised the repo and converted the source text to UTF-8. Mulong uploaded the source files and the four chapter recordings. The scripts, API routes, lib and components come to 2,841 lines.

### Segmenting the Text

The source is a single third-party file with the classical text and a modern-Chinese translation for each of the 13 chapters, plus editorial annotations and a promotional footer. `prepare-chapters.ts` finds each chapter by its header, separates the classical block from the translation block, strips annotations, and splits paragraphs into sentences after 。, ！ and ？.

```ts
function splitSentences(paragraph: string): string[] {
  const cleaned = cleanLine(paragraph);
  if (!cleaned) return [];
  // Keep punctuation attached, split after 。！？
  const parts = cleaned
    .split(/(?<=[。！？])/)
    .map((p) => p.trim())
    .filter(Boolean);
  return parts.length ? parts : [cleaned];
}
```

Each sentence keeps its classical text in `zh` and the matching modern-Chinese translation from the source in `gloss`.

### Tagging Idioms

Idioms are a hand-written list of 28 entries in `idioms.ts`. Each has the term, pinyin, a literal character-by-character reading, a modern meaning, an optional origin, and a kind: `chengyu`, `reference` or `concept`. Coverage is selective on purpose. Tagging is a substring match of each term against the sentence, so one sentence can carry several tags.

```ts
{
  term: "知己知彼",
  pinyin: "zhī jǐ zhī bǐ",
  literal: "know-self know-other",
  kind: "chengyu",
  // meaning and origin omitted here
},
```

### Choosing Audio or Speech Synthesis

`usePlayer` decides at load time whether real audio exists by asking for the file.

```ts
const head = await fetch(chapter.audioSrc, { method: "HEAD" });
if (head.ok && head.headers.get("content-type")?.includes("audio")) {
  const a = new Audio(chapter.audioSrc);
  // ...
}
```

Dropping an MP3 into `public/audio` is all it takes to switch a chapter from speech synthesis to real narration.

### The Enrichment Script

`enrich-translations.ts` sends one batch per chapter to `claude-haiku-4-5` with a system prompt asking for a JSON array of modern English translations, one per input sentence. It strips a code fence if the model added one, and it checks that the array length equals the sentence count, throwing if not. It skips chapters where every sentence already has English, so it is safe to re-run.

### The Narration Script

`generate-audio.ts` makes one ElevenLabs call per sentence, so it knows each sentence's real duration, then concatenates the MP3 chunks by raw bytes. The defaults are the `eleven_multilingual_v2` model, `mp3_44100_128` output, stability 0.45 and similarity 0.7. It writes each sentence's `start` and `end` from the measured durations, using the frame walk above.

```ts
chunks.push(result.audio);
s.start = +cursor.toFixed(2);
s.end = +(cursor + result.durationSec).toFixed(2);
cursor += result.durationSec + SENTENCE_GAP_MS / 1000;
```

It avoids ffmpeg entirely, and per-sentence calls make the alignment exact by construction.

### The Check-in Loop

The player pauses after `checkinIntervalMin` minutes of playback, three by default, and opens `CheckinModal`. The modal asks `/api/question` for a question, its rubric and a one-sentence recap, using the last 12 sentences as context. It reads the question aloud and takes the answer from the browser's speech recognition<sup>[[4]](#ref-4)</sup>, or as typed text when recognition isn't available. `/api/grade` then sends the last 10 sentences, the rubric and the answer to Sonnet 4.5 and gets back one of four verdicts, `great`, `good`, `partial` or `off`, with feedback under 30 words and a one-sentence model answer.

### The Vault and the PWA

Saved words are stored client-side with `idb-keyval` under a `vault:` prefix, so the vault works offline and never leaves the device. The service worker caches a small shell and serves stale-while-revalidate for same-origin GETs, skipping `/api/`.

## Problems

### 1. Pairing Classical Sentences With Their Translation

Each classical paragraph comes with a modern-Chinese translation paragraph, and the two don't always split into the same number of sentences. The script pairs paragraphs by index. **When the sentence counts match they align one to one; when they don't, the translation sentences are distributed by index ratio and leftovers fold into the last sentence.** The pairing is only as good as the punctuation in the source, but it keeps every sentence attached to some translation without hand alignment.

### 2. Timing Every Sentence Without an Audio Toolchain

A synced transcript needs the start and end of every sentence inside a chapter-long recording. **Narrating one sentence per call makes the boundaries known by construction, and walking the MP3 frame headers gets each clip's length without ffmpeg.** Concatenating the chunks by raw bytes then gives the chapter file, and the timestamps fall out of a running cursor.

### 3. Grading a Spoken Answer Kindly

A check-in that makes a heritage speaker feel stupid defeats the point of the app. The grading prompt tells the model to be warm and never shame, and to treat the transcription as possibly wrong, since browser speech recognition mishears. **I split the models by difficulty: question writing is short and forgiving, so it goes to Haiku 4.5; grading a free-form spoken answer against a rubric is the judgment step, so it goes to Sonnet 4.5.**

## Experiments

Two measurements needed no paid keys.

**Segmentation.** I ran `npx tsx scripts/prepare-chapters.ts`, which needs no keys, and counted sentences and idiom tags per chapter with `scripts/measure.py`.

**Narration.** I ran `ffprobe` on the four chapter MP3s to read their durations and format. The same script reports file sizes.

## Results

**Segmentation is deterministic and gives 13 chapters, 290 sentences and 169 idiom tags, or 0.583 tags per sentence.**

<figure data-figure="chart:projects/listenting/listenting-chapters"></figure>

Chapter 11 is the longest at 48 sentences and chapter 8 the shortest at 9. **Tags run from 5 in chapter 9 to 26 in chapters 1 and 11, which reflects the hand-picked list of 28 entries more than how idiomatic a chapter is.**

**The four narrated chapters come to 944.33 s of audio**: 230.43, 213.84, 278.18 and 221.88 seconds for chapters 1 to 4. The files are 3,722,279, 3,456,875, 4,486,309 and 3,585,607 bytes, about 15.2 MB together, all mono at 44.1 kHz and roughly 129 kbps. Their embedded content credentials<sup>[[5]](#ref-5)</sup> name Eleven Labs as the generator, first as created by a trained algorithm and then as edited into a composite. Chapters 5 to 13 play through browser speech synthesis.

None of this measures whether the product teaches anything. Translation quality, grading accuracy, learning outcomes, cost and latency were outside what a weekend could measure.

## What I Would Change

### Evaluate the Grader

Grading a spoken answer against a rubric is the part that most needs evidence. A small hand-labelled set of question, rubric, answer and expected verdict, run through `/api/grade`, would give an agreement rate. Speech recognition errors are a confound, so I would start with typed answers.

### Measure Cost per Session

I would log token and character counts per session and compute the cost from the price list, so the case for an offline pipeline rests on a number.

## References

1. <span id="ref-1"></span>Sun Tzu (trans. Lionel Giles). *The Art of War*. Project Gutenberg, ebook 132. [link](https://www.gutenberg.org/ebooks/132)
2. <span id="ref-2"></span>Valdés, G. *Heritage Language Students: Profiles and Possibilities*. In Peyton, Ranard and McGinnis (eds.), *Heritage Languages in America: Preserving a National Resource*, Center for Applied Linguistics and Delta Systems, 2001.
3. <span id="ref-3"></span>Krashen, S. *Principles and Practice in Second Language Acquisition*. Pergamon Press, 1982.
4. <span id="ref-4"></span>*Web Speech API*. W3C draft specification. [link](https://webaudio.github.io/web-speech-api/)
5. <span id="ref-5"></span>Coalition for Content Provenance and Authenticity. *C2PA Technical Specification*. [link](https://c2pa.org/specifications/)
6. <span id="ref-6"></span>ElevenLabs. *Text to Speech API documentation*. [link](https://elevenlabs.io/docs)
