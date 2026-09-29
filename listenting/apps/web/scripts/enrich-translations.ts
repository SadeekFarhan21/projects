/**
 * One-shot enrichment: read each chapter JSON, send sentences in a single batch
 * to Claude, and write back with `en` populated.
 *
 * Run: ANTHROPIC_API_KEY=... pnpm tsx scripts/enrich-translations.ts
 *      (optionally pass chapter ids: ... enrich-translations.ts 1 2 3)
 */
import Anthropic from "@anthropic-ai/sdk";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Chapter } from "../src/lib/types";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CHAPTERS_DIR = join(__dirname, "..", "public", "chapters");

const SYSTEM = `You translate Sun Tzu's Art of War (孙子兵法) from classical Chinese into clear, modern English.

Rules:
- Output ONLY a JSON array of strings, one per input sentence, same length and order.
- Translate each sentence faithfully but in natural modern English a learner can understand.
- Keep proper nouns (Sun Tzu, Wu, Yue, etc.) as English.
- Do not add commentary or numbering inside the strings.
- Keep idioms recognizable (e.g. 知己知彼 → "know yourself and know your enemy").`;

async function translateChapter(client: Anthropic, ch: Chapter): Promise<string[]> {
  const userInput = ch.sentences.map((s, i) => `${i + 1}. ${s.zh}`).join("\n");
  const resp = await client.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 4000,
    system: SYSTEM,
    messages: [
      {
        role: "user",
        content: `Chapter ${ch.id} — ${ch.titleEn} (${ch.titleZh}). Translate each numbered sentence into modern English. Reply with a JSON array of ${ch.sentences.length} strings.\n\n${userInput}`,
      },
    ],
  });
  const text = resp.content
    .filter((b): b is Extract<typeof b, { type: "text" }> => b.type === "text")
    .map((b) => b.text)
    .join("")
    .trim();
  // Strip code fence if any
  const cleaned = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const arr = JSON.parse(cleaned);
  if (!Array.isArray(arr) || arr.length !== ch.sentences.length) {
    throw new Error(
      `Bad translation array for ch${ch.id}: got ${Array.isArray(arr) ? arr.length : "non-array"}, expected ${ch.sentences.length}`,
    );
  }
  return arr.map(String);
}

async function main() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    console.error("ANTHROPIC_API_KEY not set — skipping enrichment.");
    process.exit(1);
  }
  const client = new Anthropic({ apiKey });

  const argIds = process.argv.slice(2);
  const ids =
    argIds.length > 0
      ? argIds
      : readdirSync(CHAPTERS_DIR)
          .filter((f) => /^\d+\.json$/.test(f))
          .map((f) => f.replace(".json", ""))
          .sort((a, b) => Number(a) - Number(b));

  for (const id of ids) {
    const path = join(CHAPTERS_DIR, `${id}.json`);
    const ch: Chapter = JSON.parse(readFileSync(path, "utf8"));
    const alreadyDone = ch.sentences.every((s) => s.en && s.en.length > 0);
    if (alreadyDone) {
      console.log(`ch${id} — already enriched, skipping`);
      continue;
    }
    process.stdout.write(`ch${id} ${ch.titleZh} (${ch.titleEn}) — translating ${ch.sentences.length} sentences... `);
    try {
      const english = await translateChapter(client, ch);
      ch.sentences.forEach((s, i) => {
        s.en = english[i];
      });
      writeFileSync(path, JSON.stringify(ch, null, 2));
      console.log("done");
    } catch (e) {
      console.error(`failed: ${(e as Error).message}`);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
