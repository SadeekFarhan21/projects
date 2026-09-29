/**
 * Generate chapter MP3s with ElevenLabs and align sentence timestamps to the
 * real audio.
 *
 * Strategy: one TTS call per sentence (so we know exact durations from the
 * returned audio), then concatenate the MP3 chunks back-to-back. ElevenLabs
 * MP3 streams are independently decodable, so naive byte concatenation plays
 * cleanly in browsers — no ffmpeg required.
 *
 * Run:
 *   ELEVENLABS_API_KEY=... pnpm tsx scripts/generate-audio.ts                # all chapters
 *   ELEVENLABS_API_KEY=... pnpm tsx scripts/generate-audio.ts 1 2 3          # specific chapters
 *   ELEVENLABS_VOICE_ID=<id> ELEVENLABS_MODEL_ID=eleven_multilingual_v2 ...  # override defaults
 *
 * After running, the chapter JSON is updated in-place:
 *   - hasAudio: true
 *   - duration: from real audio
 *   - sentences[*].start / .end: from per-sentence rendered durations
 */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { Buffer } from "node:buffer";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Chapter } from "../src/lib/types";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CHAPTERS_DIR = join(__dirname, "..", "public", "chapters");
const AUDIO_DIR = join(__dirname, "..", "public", "audio");

const DEFAULT_VOICE = process.env.ELEVENLABS_VOICE_ID_ZH || process.env.ELEVENLABS_VOICE_ID || "XB0fDUnXU5powFXDhCwa"; // Charlotte (multilingual)
const MODEL_ID = process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2";
const SENTENCE_GAP_MS = 350;
const SAMPLE_RATE = 44100;
const MP3_FORMAT = "mp3_44100_128";

interface TtsResult {
  audio: Buffer;
  /** Approx duration in seconds, parsed from MP3 frame headers. */
  durationSec: number;
}

async function ttsSentence(apiKey: string, text: string, voiceId: string): Promise<TtsResult> {
  const r = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=${MP3_FORMAT}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text,
        model_id: MODEL_ID,
        voice_settings: { stability: 0.45, similarity_boost: 0.7, style: 0.0, use_speaker_boost: true },
      }),
    },
  );
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`ElevenLabs ${r.status}: ${detail}`);
  }
  const audio = Buffer.from(await r.arrayBuffer());
  const durationSec = estimateMp3Duration(audio);
  return { audio, durationSec };
}

/** Count CBR MP3 frames and convert to seconds. Tolerates ID3v2 header. */
function estimateMp3Duration(buf: Buffer): number {
  let pos = 0;
  // Skip ID3v2 tag
  if (buf.length > 10 && buf[0] === 0x49 && buf[1] === 0x44 && buf[2] === 0x33) {
    const size = ((buf[6] & 0x7f) << 21) | ((buf[7] & 0x7f) << 14) | ((buf[8] & 0x7f) << 7) | (buf[9] & 0x7f);
    pos = 10 + size;
  }
  const BITRATES_V1L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
  const SAMPLE_RATES_V1 = [44100, 48000, 32000];
  let samples = 0;
  let sampleRate = SAMPLE_RATE;
  while (pos + 4 <= buf.length) {
    const b1 = buf[pos];
    const b2 = buf[pos + 1];
    const b3 = buf[pos + 2];
    if (b1 !== 0xff || (b2 & 0xe0) !== 0xe0) {
      pos++;
      continue;
    }
    const version = (b2 >> 3) & 0x3; // 3 = MPEG1, 2 = MPEG2, 0 = MPEG2.5
    const layer = (b2 >> 1) & 0x3;   // 1 = Layer III
    const bitrateIdx = (b3 >> 4) & 0xf;
    const sampleRateIdx = (b3 >> 2) & 0x3;
    const padding = (b3 >> 1) & 0x1;
    if (layer !== 1 || bitrateIdx === 0 || bitrateIdx === 15 || sampleRateIdx === 3) {
      pos++;
      continue;
    }
    let bitrateKbps: number;
    let sr: number;
    let samplesPerFrame: number;
    if (version === 3) {
      bitrateKbps = BITRATES_V1L3[bitrateIdx];
      sr = SAMPLE_RATES_V1[sampleRateIdx];
      samplesPerFrame = 1152;
    } else {
      // MPEG2/2.5 Layer III
      const V2L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
      bitrateKbps = V2L3[bitrateIdx];
      sr = version === 2 ? SAMPLE_RATES_V1[sampleRateIdx] / 2 : SAMPLE_RATES_V1[sampleRateIdx] / 4;
      samplesPerFrame = 576;
    }
    const frameLen = Math.floor((samplesPerFrame / 8 * bitrateKbps * 1000) / sr) + padding;
    if (frameLen <= 0) {
      pos++;
      continue;
    }
    samples += samplesPerFrame;
    sampleRate = sr;
    pos += frameLen;
  }
  return samples / sampleRate;
}

/** Generate `seconds` of silence as an MP3 chunk by stitching together silent frames from ElevenLabs?
 *  Simpler: encode silence directly as a minimal MP3 with the same encoder is overkill.
 *  We use an in-memory zero-PCM → MP3 by inserting tiny gap via repeated silent ElevenLabs calls is wasteful.
 *  Instead: we don't insert silence files. Sentence boundaries naturally fall on MP3 frame boundaries when concatenated.
 *  Listeners get a tiny natural breath. If you want a configurable gap, run through ffmpeg post-hoc.
 */

async function generateChapter(apiKey: string, chapter: Chapter, voiceId: string) {
  console.log(`\nch${chapter.id} ${chapter.titleZh} (${chapter.titleEn}) — ${chapter.sentences.length} sentences`);
  const chunks: Buffer[] = [];
  let cursor = 0;
  for (let i = 0; i < chapter.sentences.length; i++) {
    const s = chapter.sentences[i];
    process.stdout.write(`  [${i + 1}/${chapter.sentences.length}] ${s.zh.slice(0, 18)}… `);
    let result: TtsResult;
    try {
      result = await ttsSentence(apiKey, s.zh, voiceId);
    } catch (e) {
      console.log(`FAILED: ${(e as Error).message}`);
      throw e;
    }
    chunks.push(result.audio);
    s.start = +cursor.toFixed(2);
    s.end = +(cursor + result.durationSec).toFixed(2);
    cursor += result.durationSec + SENTENCE_GAP_MS / 1000;
    console.log(`${result.durationSec.toFixed(2)}s`);
  }
  const merged = Buffer.concat(chunks);
  mkdirSync(AUDIO_DIR, { recursive: true });
  const outPath = join(AUDIO_DIR, `ch${chapter.id}.mp3`);
  writeFileSync(outPath, merged);
  chapter.duration = +cursor.toFixed(2);
  chapter.hasAudio = true;
  chapter.audioSrc = `/audio/ch${chapter.id}.mp3`;
  writeFileSync(join(CHAPTERS_DIR, `${chapter.id}.json`), JSON.stringify(chapter, null, 2));
  console.log(`  → wrote ${outPath} (${(merged.length / 1024).toFixed(0)} KB, ${chapter.duration.toFixed(1)}s)`);
}

async function main() {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    console.error("ELEVENLABS_API_KEY not set.");
    process.exit(1);
  }
  const voiceId = DEFAULT_VOICE;
  const argIds = process.argv.slice(2);
  const ids =
    argIds.length > 0
      ? argIds
      : readdirSync(CHAPTERS_DIR)
          .filter((f) => /^\d+\.json$/.test(f))
          .map((f) => f.replace(".json", ""))
          .sort((a, b) => Number(a) - Number(b));

  // Refresh index after we update each chapter
  const indexPath = join(CHAPTERS_DIR, "index.json");
  const index = existsSync(indexPath)
    ? (JSON.parse(readFileSync(indexPath, "utf8")) as { chapters: any[] })
    : { chapters: [] };

  for (const id of ids) {
    const path = join(CHAPTERS_DIR, `${id}.json`);
    const ch: Chapter = JSON.parse(readFileSync(path, "utf8"));
    await generateChapter(apiKey, ch, voiceId);
    const meta = index.chapters.find((c) => c.id === id);
    if (meta) {
      meta.duration = ch.duration;
      meta.hasAudio = true;
    }
  }
  writeFileSync(indexPath, JSON.stringify(index, null, 2));
  console.log("\nDone.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
