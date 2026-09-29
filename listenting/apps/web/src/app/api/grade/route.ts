import { NextResponse } from "next/server";
import { getAnthropic, MODELS, stripFence, textOf } from "@/lib/anthropic";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Body: {
 *   question, rubric, contextSummary,
 *   contextSentences: Array<{ zh, en?, gloss? }>,
 *   answer: string,         // transcribed user answer
 *   answerLanguage: "en" | "zh" | "mixed"
 * }
 * Response: { verdict: "great"|"good"|"partial"|"off", feedback, modelAnswer }
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const question = String(body?.question ?? "").trim();
    const rubric = String(body?.rubric ?? "").trim();
    const contextSummary = String(body?.contextSummary ?? "").trim();
    const sentences: Array<{ zh: string; en?: string; gloss?: string }> = body?.contextSentences ?? [];
    const answer = String(body?.answer ?? "").trim();
    if (!question || !answer) {
      return NextResponse.json({ error: "missing-fields" }, { status: 400 });
    }
    const context = sentences
      .slice(-10)
      .map((s, i) => `${i + 1}. ${s.zh}${s.en ? `  → ${s.en}` : s.gloss ? `  (${s.gloss})` : ""}`)
      .join("\n");

    const sys = `You are a warm, encouraging Chinese listening tutor grading a single spoken answer.

Return JSON only: {"verdict":"great|good|partial|off","feedback":"<= 30 words, encouraging, second-person","modelAnswer":"a one-sentence reference answer in English"}

Grading scale:
- great:  answers the question fully and accurately
- good:   correct but missing some nuance
- partial: partly right or partly understood; show kindness
- off:    misunderstood — gently redirect, don't shame`;

    const client = getAnthropic();
    const resp = await client.messages.create({
      model: MODELS.smart,
      max_tokens: 400,
      system: sys,
      messages: [
        {
          role: "user",
          content: `Story context (recent sentences):
${context}

Quick recap: ${contextSummary}

Question I asked: ${question}
Rubric for "correct": ${rubric || "(none — judge by the story)"}

The student replied (auto-transcribed, may have errors): "${answer}"

Grade now.`,
        },
      ],
    });
    const raw = stripFence(textOf(resp));
    let parsed: { verdict?: string; feedback?: string; modelAnswer?: string };
    try {
      parsed = JSON.parse(raw);
    } catch {
      return NextResponse.json({ error: "bad-model-response", raw }, { status: 502 });
    }
    const verdict = ["great", "good", "partial", "off"].includes(String(parsed.verdict))
      ? (parsed.verdict as "great" | "good" | "partial" | "off")
      : "partial";
    return NextResponse.json({
      verdict,
      feedback: parsed.feedback ?? "Nice try — keep going.",
      modelAnswer: parsed.modelAnswer ?? "",
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
