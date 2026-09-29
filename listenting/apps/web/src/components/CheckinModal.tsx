"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Chapter, Sentence, CheckinGrade, CheckinQuestion } from "@/lib/types";
import { createRecognizer, hasBrowserSTT, speak } from "@/lib/speech";
import { Mic, Close, Check, Sparkle } from "./Icons";
import { Waveform } from "./Waveform";

interface Props {
  open: boolean;
  chapter: Chapter;
  /** Sentences the user has heard so far (used as context) */
  contextSentences: Sentence[];
  language: "en" | "zh";
  onClose: (resumePlayback: boolean) => void;
}

type Phase = "loading" | "asking" | "listening" | "grading" | "result" | "error";

const VERDICT_COLOR: Record<CheckinGrade["verdict"], string> = {
  great: "text-jade",
  good: "text-jade-soft",
  partial: "text-gold",
  off: "text-sun",
};
const VERDICT_LABEL: Record<CheckinGrade["verdict"], string> = {
  great: "Nailed it",
  good: "Good",
  partial: "Almost",
  off: "Let's revisit",
};

export function CheckinModal({ open, chapter, contextSentences, language, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [question, setQuestion] = useState<CheckinQuestion | null>(null);
  const [answer, setAnswer] = useState("");
  const [interim, setInterim] = useState("");
  const [grade, setGrade] = useState<CheckinGrade | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<ReturnType<typeof createRecognizer> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const sttAvailable = useMemo(() => hasBrowserSTT(), []);

  /* ---------- Lifecycle ---------- */
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPhase("loading");
    setQuestion(null);
    setAnswer("");
    setInterim("");
    setGrade(null);
    setError(null);

    (async () => {
      try {
        const resp = await fetch("/api/question", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter: { id: chapter.id, titleEn: chapter.titleEn, titleZh: chapter.titleZh },
            contextSentences: contextSentences.map((s) => ({ zh: s.zh, gloss: s.gloss, en: s.en })),
            language,
          }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err?.error || `status ${resp.status}`);
        }
        const data = (await resp.json()) as CheckinQuestion;
        if (cancelled) return;
        setQuestion(data);
        setPhase("asking");

        // Try ElevenLabs first; fall back to browser TTS
        const spoken = await tryElevenLabs(data.question, language);
        if (cancelled) return;
        if (!spoken) await speak(data.question, language);
        if (cancelled) return;
        setPhase("listening");
        startListening();
      } catch (e) {
        if (cancelled) return;
        setError((e as Error).message);
        setPhase("error");
      }
    })();

    return () => {
      cancelled = true;
      recRef.current?.stop();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const tryElevenLabs = async (text: string, lang: "en" | "zh"): Promise<boolean> => {
    try {
      const r = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language: lang }),
      });
      if (!r.ok) return false;
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = new Audio(url);
      audioRef.current = a;
      await new Promise<void>((resolve) => {
        a.onended = () => resolve();
        a.onerror = () => resolve();
        a.play().catch(() => resolve());
      });
      URL.revokeObjectURL(url);
      return true;
    } catch {
      return false;
    }
  };

  const startListening = useCallback(() => {
    if (!sttAvailable) {
      // Tap to type instead — phase remains "listening", text input shown
      return;
    }
    const rec = createRecognizer(language === "zh" ? "zh-CN" : "en-US");
    if (!rec) return;
    recRef.current = rec;
    rec.start(
      (text) => setInterim(text),
      (final) => setAnswer((a) => (a ? `${a} ${final}` : final).trim()),
      () => {
        // ended naturally
      },
      (err) => {
        if (err !== "no-speech" && err !== "aborted") {
          setError(err);
        }
      },
    );
  }, [language, sttAvailable]);

  const stopListening = useCallback(() => {
    recRef.current?.stop();
  }, []);

  const submit = useCallback(
    async (finalText?: string) => {
      const txt = (finalText ?? answer ?? interim).trim();
      if (!txt || !question) return;
      stopListening();
      setPhase("grading");
      try {
        const r = await fetch("/api/grade", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.question,
            rubric: question.rubric,
            contextSummary: question.contextSummary,
            contextSentences: contextSentences.map((s) => ({ zh: s.zh, en: s.en, gloss: s.gloss })),
            answer: txt,
            answerLanguage: language,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err?.error || `status ${r.status}`);
        }
        const data = (await r.json()) as CheckinGrade;
        setGrade(data);
        setPhase("result");
      } catch (e) {
        setError((e as Error).message);
        setPhase("error");
      }
    },
    [answer, interim, question, contextSentences, language, stopListening],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-ink/85 backdrop-blur-md flex flex-col pb-safe">
      <button
        onClick={() => onClose(false)}
        className="absolute top-5 right-5 text-ink-200 hover:text-white p-2"
        aria-label="Close"
      >
        <Close width={24} height={24} />
      </button>

      <div className="flex-1 flex flex-col items-center justify-center px-6 max-w-xl mx-auto w-full">
        {phase === "loading" && (
          <div className="text-ink-200 flex flex-col items-center gap-3 animate-fade-in">
            <Sparkle className="text-gold animate-pulse-soft" width={28} height={28} />
            <p className="text-sm">Thinking of a good question…</p>
          </div>
        )}

        {phase === "asking" && question && (
          <div className="text-center animate-fade-in">
            <div className="text-xs uppercase tracking-widest text-gold mb-3">Listen</div>
            <p className="text-2xl sm:text-3xl text-white font-serif-display leading-snug">{question.question}</p>
            <div className="mt-6 text-ink-200">
              <Waveform active count={7} className="text-gold" />
            </div>
          </div>
        )}

        {phase === "listening" && question && (
          <div className="w-full animate-fade-in">
            <div className="text-xs uppercase tracking-widest text-gold mb-3 text-center">Your turn</div>
            <p className="text-xl sm:text-2xl text-white font-serif-display leading-snug text-center mb-8">
              {question.question}
            </p>

            <div className="ink-card-dark p-5 min-h-[120px] flex items-center justify-center text-center mb-6">
              <p className="text-lg text-ink-50">
                {interim || answer || (
                  <span className="text-ink-300">
                    {sttAvailable ? "Speak your answer in any language…" : "Type your answer below."}
                  </span>
                )}
              </p>
            </div>

            {!sttAvailable && (
              <input
                autoFocus
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") submit(answer);
                }}
                placeholder="Type and press Enter"
                className="w-full mb-4 rounded-xl border border-white/20 bg-white/5 px-4 py-3 text-white placeholder:text-ink-300 focus:outline-none focus:border-gold"
              />
            )}

            <div className="flex items-center justify-center gap-3">
              {sttAvailable && (
                <button
                  onClick={() => {
                    stopListening();
                    submit();
                  }}
                  className="btn-sun px-7 py-4"
                >
                  <Mic width={20} height={20} />
                  Done
                </button>
              )}
              {!sttAvailable && (
                <button
                  onClick={() => submit(answer)}
                  disabled={!answer.trim()}
                  className="btn-sun px-7 py-4 disabled:opacity-40"
                >
                  Submit
                </button>
              )}
              <button onClick={() => onClose(true)} className="btn-ghost text-ink-200 hover:text-white hover:bg-white/10">
                Skip
              </button>
            </div>
          </div>
        )}

        {phase === "grading" && (
          <div className="text-ink-200 flex flex-col items-center gap-3 animate-fade-in">
            <Sparkle className="text-gold animate-pulse-soft" width={28} height={28} />
            <p className="text-sm">Grading your answer…</p>
          </div>
        )}

        {phase === "result" && grade && (
          <div className="text-center w-full animate-fade-in">
            <div className={`text-xs uppercase tracking-widest mb-3 ${VERDICT_COLOR[grade.verdict]}`}>
              {VERDICT_LABEL[grade.verdict]}
            </div>
            <p className="text-xl sm:text-2xl text-white leading-snug mb-6">{grade.feedback}</p>
            {grade.modelAnswer && (
              <div className="ink-card-dark p-4 text-sm text-ink-100/90 mb-8 text-left">
                <div className="text-xs uppercase tracking-wider text-ink-300 mb-1">Reference answer</div>
                {grade.modelAnswer}
              </div>
            )}
            <button onClick={() => onClose(true)} className="btn-sun w-full max-w-xs mx-auto">
              <Check width={18} height={18} />
              Resume listening
            </button>
          </div>
        )}

        {phase === "error" && (
          <div className="text-center max-w-sm animate-fade-in">
            <p className="text-sm text-sun mb-3">Couldn't run the check-in.</p>
            <p className="text-ink-200 text-sm mb-6">{error}</p>
            <button onClick={() => onClose(true)} className="btn-ghost text-ink-100 hover:text-white hover:bg-white/10">
              Skip and keep listening
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
