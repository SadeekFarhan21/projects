import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Fallback STT for browsers without the Web Speech API (Safari iOS standalone PWA, mostly).
 * POST multipart/form-data with field `audio` (webm/opus or m4a/aac).
 * Uses ElevenLabs Speech-to-Text if available.
 */
export async function POST(req: Request) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "no-stt-provider" }, { status: 501 });
  }
  try {
    const form = await req.formData();
    const audio = form.get("audio");
    if (!(audio instanceof Blob)) {
      return NextResponse.json({ error: "missing-audio" }, { status: 400 });
    }
    const upstream = new FormData();
    upstream.append("file", audio, "speech.webm");
    upstream.append("model_id", "scribe_v1");

    const resp = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
      method: "POST",
      headers: { "xi-api-key": apiKey },
      body: upstream,
    });
    if (!resp.ok) {
      const text = await resp.text();
      return NextResponse.json({ error: "stt-failed", detail: text }, { status: 502 });
    }
    const data = (await resp.json()) as { text?: string; language_code?: string };
    return NextResponse.json({ text: data.text ?? "", language: data.language_code ?? null });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
