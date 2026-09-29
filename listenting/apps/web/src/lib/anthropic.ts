import Anthropic from "@anthropic-ai/sdk";

let _client: Anthropic | null = null;

export function getAnthropic(): Anthropic {
  if (_client) return _client;
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");
  _client = new Anthropic({ apiKey });
  return _client;
}

export const MODELS = {
  // Fast, cheap — for translation and question generation.
  fast: "claude-haiku-4-5" as const,
  // Higher quality — for grading.
  smart: "claude-sonnet-4-5" as const,
};

/** Extract concatenated text from a Messages response. */
export function textOf(resp: Anthropic.Messages.Message): string {
  return resp.content
    .filter((b): b is Anthropic.Messages.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("")
    .trim();
}

/** Strip a fenced code block (```json ... ```) if present. */
export function stripFence(s: string): string {
  return s.replace(/^```(?:json|JSON)?\s*/i, "").replace(/```\s*$/i, "").trim();
}
