/**
 * Reads the combined source file `chinese/孙子兵法_原文与译文.txt`,
 * splits it into 13 chapters, segments sentences, aligns translation
 * paragraphs, tags idioms, generates synthetic timestamps based on
 * Chinese character count (≈ 4.5 chars/sec for narrated audio), and
 * writes one JSON per chapter plus an index.
 *
 * Run: pnpm prepare-chapters
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Chapter, ChapterIndex, Sentence } from "../src/lib/types";
import { CHAPTER_META } from "./translations";
import { findIdiomsIn } from "./idioms";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "../../..");
const SRC = join(ROOT, "chinese", "孙子兵法_原文与译文.txt");
const OUT_DIR = join(__dirname, "..", "public", "chapters");

const CHARS_PER_SECOND = 4.5; // narration pace
const SENTENCE_GAP = 0.35;    // pause between sentences

const CHAPTER_HEADERS: Array<{ id: string; header: RegExp; translation: RegExp }> = [
  { id: "1",  header: /^始计第一$/,   translation: /^始计篇 译文$/ },
  { id: "2",  header: /^作战第二$/,   translation: /^作战篇 译文$/ },
  { id: "3",  header: /^谋攻第三$/,   translation: /^谋攻篇 译文$/ },
  { id: "4",  header: /^军形第四$/,   translation: /^军形篇 译文$/ },
  { id: "5",  header: /^兵势第五$/,   translation: /^兵势篇 译文$/ },
  { id: "6",  header: /^虚实第六$/,   translation: /^虚实篇 译文$/ },
  { id: "7",  header: /^军争第七$/,   translation: /^军争篇 译文$/ },
  { id: "8",  header: /^九变第八$/,   translation: /^九变篇 译文$/ },
  { id: "9",  header: /^行军第九$/,   translation: /^行军篇 译文$/ },
  { id: "10", header: /^地形第十$/,   translation: /^地形篇 译文$/ },
  { id: "11", header: /^九地第十一$/, translation: /^九地篇 译文$/ },
  { id: "12", header: /^火攻第十二$/, translation: /^火攻篇 译文$/ },
  { id: "13", header: /^用间第十三$/, translation: /^用间篇 译文$/ },
];

/** Strip editorial annotation lines like `【注：】①：...` */
function cleanLine(s: string): string {
  return s
    .replace(/【[^】]*】[^]*$/g, "")
    .replace(/□[①②③④⑤⑥⑦⑧⑨⑩]/g, "")
    .replace(/\([^)]*\)/g, "")
    .trim();
}

/** Split a Chinese paragraph into sentences by terminal punctuation. */
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

function loadChaptersFromSource(): Array<{ id: string; origParagraphs: string[]; translParagraphs: string[] }> {
  const raw = readFileSync(SRC, "utf8");
  const lines = raw.split(/\r?\n/);
  const out: Array<{ id: string; origParagraphs: string[]; translParagraphs: string[] }> = [];

  for (let i = 0; i < CHAPTER_HEADERS.length; i++) {
    const cur = CHAPTER_HEADERS[i];
    const next = CHAPTER_HEADERS[i + 1];
    const headerIdx = lines.findIndex((l) => cur.header.test(l.trim()));
    if (headerIdx === -1) throw new Error(`Could not locate chapter ${cur.id} header`);
    const endIdx = next ? lines.findIndex((l) => next.header.test(l.trim())) : lines.length;
    if (next && endIdx === -1) throw new Error(`Could not locate next chapter for ${cur.id}`);

    // Translation block normally starts with a "X篇 译文" marker between header and endIdx.
    let translIdx = lines.findIndex((l, idx) => idx > headerIdx && idx < endIdx && cur.translation.test(l.trim()));
    let translStartLine = translIdx === -1 ? -1 : translIdx + 1;

    // Ch13 has no marker — translation paragraphs simply begin where classical
    // "孙子曰" ends and modern "孙子说" begins. Fall back to that.
    if (translIdx === -1) {
      const fallback = lines.findIndex(
        (l, idx) => idx > headerIdx && idx < endIdx && /^孙子说[：:]/.test(l.trim()),
      );
      if (fallback === -1) throw new Error(`Could not locate translation block for chapter ${cur.id}`);
      translIdx = fallback;
      translStartLine = fallback; // include the "孙子说" line itself
    }

    const origLines = lines.slice(headerIdx + 1, translIdx).map((l) => l.trim()).filter(Boolean);
    const translLines = lines
      .slice(translStartLine, endIdx)
      .map((l) => l.trim())
      .filter(Boolean)
      // Drop the trailing "更多精彩好书..." promo footer
      .filter((l) => !l.includes("更多精彩好书") && !l.includes("qinkan.net"));

    out.push({ id: cur.id, origParagraphs: origLines, translParagraphs: translLines });
  }
  return out;
}

function alignParagraphSentences(
  origPara: string,
  translPara: string,
  startTime: number,
): { sentences: Sentence[]; nextTime: number; counter: number } {
  // Caller will offset the counter; we'll return relative sentences and let caller renumber.
  const origSents = splitSentences(origPara);
  const translSents = splitSentences(translPara);

  // If both have same count, 1:1 align. Otherwise fold extra translation tail into the last sentence.
  const result: Sentence[] = [];
  let t = startTime;
  const len = Math.max(origSents.length, 1);

  for (let i = 0; i < len; i++) {
    const zh = origSents[i] ?? "";
    let en: string;
    if (origSents.length === translSents.length) {
      en = translSents[i] ?? "";
    } else if (origSents.length === 1) {
      en = translSents.join(" ");
    } else if (translSents.length === 1) {
      en = i === 0 ? translSents[0] : "";
    } else {
      // Best-effort: distribute by index ratio, fold leftovers into last
      const ratio = translSents.length / origSents.length;
      const startIdx = Math.floor(i * ratio);
      const endIdx = i === origSents.length - 1 ? translSents.length : Math.floor((i + 1) * ratio);
      en = translSents.slice(startIdx, Math.max(startIdx + 1, endIdx)).join(" ");
    }

    const charCount = zh.replace(/[，。！？、：；""''（）()]/g, "").length;
    const dur = Math.max(1.2, charCount / CHARS_PER_SECOND);
    const sent: Sentence = {
      id: i, // renumbered by caller
      zh,
      gloss: en.trim(),
      start: +t.toFixed(2),
      end: +(t + dur).toFixed(2),
      idioms: findIdiomsIn(zh),
    };
    if (!sent.idioms?.length) delete sent.idioms;
    result.push(sent);
    t += dur + SENTENCE_GAP;
  }
  return { sentences: result, nextTime: t, counter: 0 };
}

function buildChapter(id: string, origParagraphs: string[], translParagraphs: string[]): Chapter {
  const meta = CHAPTER_META[id];
  // Pair paragraphs by index; pad shorter side with empty strings
  const n = Math.max(origParagraphs.length, translParagraphs.length);
  const all: Sentence[] = [];
  let t = 0;
  let counter = 0;
  for (let i = 0; i < n; i++) {
    const o = origParagraphs[i] ?? "";
    const en = translParagraphs[i] ?? "";
    if (!o) continue;
    const { sentences, nextTime } = alignParagraphSentences(o, en, t);
    for (const s of sentences) {
      s.id = counter++;
      all.push(s);
    }
    t = nextTime + 0.6; // paragraph gap
  }

  return {
    id,
    titleZh: meta.titleZh,
    titlePinyin: meta.titlePinyin,
    titleEn: meta.titleEn,
    duration: +t.toFixed(2),
    audioSrc: `/audio/ch${id}.mp3`,
    hasAudio: false, // flipped to true at runtime if the file exists; player falls back to TTS otherwise
    sentences: all,
  };
}

function main() {
  mkdirSync(OUT_DIR, { recursive: true });
  const chapters = loadChaptersFromSource();
  const index: ChapterIndex = { chapters: [] };

  for (const c of chapters) {
    const built = buildChapter(c.id, c.origParagraphs, c.translParagraphs);
    writeFileSync(join(OUT_DIR, `${c.id}.json`), JSON.stringify(built, null, 2));
    const idiomCount = built.sentences.reduce((n, s) => n + (s.idioms?.length ?? 0), 0);
    console.log(
      `ch${c.id} ${built.titleZh} (${built.titleEn}) — ${built.sentences.length} sentences, ${idiomCount} idioms, ~${Math.round(built.duration / 60)}m`,
    );
    index.chapters.push({
      id: built.id,
      titleZh: built.titleZh,
      titleEn: built.titleEn,
      titlePinyin: built.titlePinyin,
      duration: built.duration,
      hasAudio: built.hasAudio,
      sentenceCount: built.sentences.length,
    });
  }

  writeFileSync(join(OUT_DIR, "index.json"), JSON.stringify(index, null, 2));
  console.log(`\nWrote ${chapters.length} chapters to ${OUT_DIR}`);
}

main();
