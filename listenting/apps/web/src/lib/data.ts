import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Chapter, ChapterIndex } from "./types";

const PUBLIC_DIR = join(process.cwd(), "public", "chapters");

export async function getIndex(): Promise<ChapterIndex> {
  const raw = await readFile(join(PUBLIC_DIR, "index.json"), "utf8");
  return JSON.parse(raw) as ChapterIndex;
}

export async function getChapter(id: string): Promise<Chapter | null> {
  try {
    const raw = await readFile(join(PUBLIC_DIR, `${id}.json`), "utf8");
    return JSON.parse(raw) as Chapter;
  } catch {
    return null;
  }
}
