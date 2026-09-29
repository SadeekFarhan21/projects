import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * POST { text: string, voice?: string, language?: "en" | "zh" }
 * → audio/mpeg stream (ElevenLabs). Falls back to 501 if no key, so the
 *   client knows to use the browser's SpeechSynthesis API instead.
 */
export async function POST(req: Request) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "no-elevenlabs-key" }, { status: 501 });
  }
  try {
    const { text, voice, language } = await req.json();
    if (!text || typeof text !== "string") {
      return NextResponse.json({ error: "missing-text" }, { status: 400 });
    }
    const voiceId =
      voice ||
      (language === "zh"
        ? process.env.ELEVENLABS_VOICE_ID_ZH || "21m00Tcm4TlvDq8ikWAM"
        : process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM");

    const upstream = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=mp3_44100_128`,
      {
        method: "POST",
        headers: {
          "xi-api-key": apiKey,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: language === "zh" ? "eleven_multilingual_v2" : "eleven_turbo_v2_5",
          voice_settings: { stability: 0.5, similarity_boost: 0.7 },
        }),
      },
    );
    if (!upstream.ok) {
      const err = await upstream.text();
      return NextResponse.json({ error: "elevenlabs-failed", detail: err }, { status: 502 });
    }
    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "audio/mpeg",
        "Cache-Control": "no-store",
      },
    });
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
