import { NextResponse } from "next/server";
import { getAnthropic, MODELS, stripFence, textOf } from "@/lib/anthropic";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Body: { sentences: Array<{ id: number; zh: string }>, chapter: { id: string; titleEn: string } }
 * Response: { translations: Array<{ id: number; en: string }> }
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const sentences = Array.isArray(body?.sentences) ? body.sentences : [];
    const chapter = body?.chapter ?? {};
    if (sentences.length === 0) {
      return NextResponse.json({ translations: [] });
    }
    const numbered = sentences
      .map((s: { id: number; zh: string }, i: number) => `${i + 1}. ${s.zh}`)
      .join("\n");

    const client = getAnthropic();
    const resp = await client.messages.create({
      model: MODELS.fast,
      max_tokens: 2400,
      system:
        "Translate classical Chinese (Sun Tzu's Art of War) into clear, modern English. Output ONLY a JSON array of strings in the same order and length as the input. No commentary, no numbering inside the strings. Keep idioms recognizable.",
      messages: [
        {
          role: "user",
          content: `Chapter ${chapter.id ?? ""} ${chapter.titleEn ?? ""}. Translate each numbered sentence. Reply with a JSON array of ${sentences.length} strings.\n\n${numbered}`,
        },
      ],
    });
    const raw = stripFence(textOf(resp));
    let arr: string[] = [];
    try {
      arr = JSON.parse(raw);
    } catch {
      return NextResponse.json({ error: "bad-model-response", raw }, { status: 502 });
    }
    if (!Array.isArray(arr) || arr.length !== sentences.length) {
      return NextResponse.json({ error: "length-mismatch", got: arr.length }, { status: 502 });
    }
    return NextResponse.json({
      translations: sentences.map((s: { id: number }, i: number) => ({ id: s.id, en: String(arr[i] ?? "") })),
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
