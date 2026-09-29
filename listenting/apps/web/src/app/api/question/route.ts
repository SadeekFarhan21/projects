import { NextResponse } from "next/server";
import { getAnthropic, MODELS, stripFence, textOf } from "@/lib/anthropic";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Generate a comprehension question grounded in the transcript so far.
 *
 * Body: {
 *   chapter: { id, titleEn, titleZh },
 *   contextSentences: Array<{ zh: string, gloss?: string, en?: string }>,
 *   language: "en" | "zh"   // language for the question
 * }
 *
 * Response: { question, contextSummary, rubric }
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const chapter = body?.chapter ?? {};
    const sentences: Array<{ zh: string; gloss?: string; en?: string }> = body?.contextSentences ?? [];
    const language: "en" | "zh" = body?.language === "zh" ? "zh" : "en";

    if (sentences.length === 0) {
      return NextResponse.json({ error: "no-context" }, { status: 400 });
    }

    const lastN = sentences.slice(-12); // keep prompt small — recent beats matter most
    const context = lastN
      .map((s, i) => {
        const help = s.en || s.gloss || "";
        return `${i + 1}. ${s.zh}${help ? `  (${help})` : ""}`;
      })
      .join("\n");

    const sys =
      language === "en"
        ? `You are a warm, encouraging Chinese listening tutor. The student is listening to Sun Tzu's Art of War in Chinese.
Generate ONE short comprehension question in plain spoken English about what just happened in the story so far.
Aim for the level of an intermediate learner: ask about cause, character, or meaning — not memorized phrasing.
Keep it under 18 words. End with a question mark.

Also return a short rubric (1-2 sentences) describing what a correct answer should mention.

Output JSON: {"question": "...", "contextSummary": "one-sentence recap", "rubric": "..."}`
        : `你是一位温和的中文听力老师。学生正在听《孙子兵法》。
用简洁口语化的中文，提一个关于刚才内容的理解性问题。
难度适合中级学习者；不超过 20 字；以问号结尾。
同时给出 1-2 句简短的评分标准。

只输出 JSON: {"question":"...","contextSummary":"一句话概述","rubric":"..."}`;

    const client = getAnthropic();
    const resp = await client.messages.create({
      model: MODELS.fast,
      max_tokens: 500,
      system: sys,
      messages: [
        {
          role: "user",
          content: `Chapter ${chapter.id ?? ""} — ${chapter.titleEn ?? ""} (${chapter.titleZh ?? ""}).
The student just heard these sentences (most recent last):

${context}

Generate the JSON now.`,
        },
      ],
    });
    const raw = stripFence(textOf(resp));
    let parsed: { question?: string; contextSummary?: string; rubric?: string };
    try {
      parsed = JSON.parse(raw);
    } catch {
      return NextResponse.json({ error: "bad-model-response", raw }, { status: 502 });
    }
    if (!parsed.question) {
      return NextResponse.json({ error: "no-question", raw }, { status: 502 });
    }
    return NextResponse.json({
      question: parsed.question,
      contextSummary: parsed.contextSummary ?? "",
      rubric: parsed.rubric ?? "",
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
