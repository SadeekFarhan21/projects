"use client";

/* ----- Speech recognition (browser-native, with server fallback shape) ----- */

type SR = any;

interface SpeechRecognitionResultEvent {
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string; confidence: number };
  }>;
  resultIndex: number;
}

export function hasBrowserSTT(): boolean {
  if (typeof window === "undefined") return false;
  return "webkitSpeechRecognition" in window || "SpeechRecognition" in window;
}

export function createRecognizer(language: "en-US" | "zh-CN" | "auto" = "auto"): {
  start: (
    onInterim: (text: string) => void,
    onFinal: (text: string) => void,
    onEnd: () => void,
    onError: (err: string) => void,
  ) => void;
  stop: () => void;
} | null {
  if (typeof window === "undefined") return null;
  const Ctor: any = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!Ctor) return null;
  const r: SR = new Ctor();
  if (language !== "auto") r.lang = language;
  else r.lang = "en-US"; // we add zh-CN sibling recognizer if needed
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;

  let started = false;
  let onInterimFn: ((s: string) => void) | null = null;
  let onFinalFn: ((s: string) => void) | null = null;
  let onEndFn: (() => void) | null = null;
  let onErrFn: ((s: string) => void) | null = null;
  let finalText = "";

  r.onresult = (event: SpeechRecognitionResultEvent) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const res = event.results[i];
      const t = res[0].transcript;
      if (res.isFinal) finalText += t;
      else interim += t;
    }
    if (interim) onInterimFn?.(finalText + interim);
    else onInterimFn?.(finalText);
  };
  r.onerror = (e: any) => onErrFn?.(String(e?.error || "speech-error"));
  r.onend = () => {
    if (finalText) onFinalFn?.(finalText);
    onEndFn?.();
    started = false;
  };

  return {
    start(onInterim, onFinal, onEnd, onError) {
      onInterimFn = onInterim;
      onFinalFn = onFinal;
      onEndFn = onEnd;
      onErrFn = onError;
      finalText = "";
      try {
        r.start();
        started = true;
      } catch (e) {
        onError(String((e as Error).message));
      }
    },
    stop() {
      if (started) {
        try {
          r.stop();
        } catch {
          /* noop */
        }
      }
    },
  };
}

/* ----- Speech synthesis ----- */

export async function speak(text: string, language: "en" | "zh" = "en"): Promise<void> {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  return new Promise<void>((resolve) => {
    const synth = window.speechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = language === "zh" ? "zh-CN" : "en-US";
    u.rate = 1;
    u.pitch = 1;
    u.onend = () => resolve();
    u.onerror = () => resolve();
    synth.speak(u);
  });
}
