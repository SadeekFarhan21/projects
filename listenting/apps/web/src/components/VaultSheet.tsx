"use client";
import { useEffect, useState } from "react";
import type { Sentence, Idiom } from "@/lib/types";
import { saveEntry } from "@/lib/vault";
import { Check, Close, BookmarkSolid } from "./Icons";

interface Props {
  open: boolean;
  sentence: Sentence | null;
  chapterId: string;
  timestamp: number;
  onClose: () => void;
  onSaved?: () => void;
}

/**
 * Bottom sheet that lets the user pick a word/phrase from the active sentence
 * (or one of its tagged idioms) and save it to the vault. Saves go to IndexedDB.
 */
export function VaultSheet({ open, sentence, chapterId, timestamp, onClose, onSaved }: Props) {
  const [picked, setPicked] = useState<{ term: string; pinyin?: string; meaning?: string } | null>(null);
  const [customMeaning, setCustomMeaning] = useState("");
  const [saving, setSaving] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (open && sentence) {
      const firstIdiom = sentence.idioms?.[0];
      if (firstIdiom) {
        setPicked({ term: firstIdiom.term, pinyin: firstIdiom.pinyin, meaning: firstIdiom.meaning });
      } else {
        setPicked(null);
      }
      setCustomMeaning("");
      setDone(false);
    }
  }, [open, sentence]);

  if (!open || !sentence) return null;

  const handleSave = async () => {
    if (!picked) return;
    setSaving(true);
    try {
      await saveEntry({
        word: picked.term,
        pinyin: picked.pinyin,
        meaning: picked.meaning || customMeaning || "(no definition yet)",
        sentenceZh: sentence.zh,
        sentenceEn: sentence.en || sentence.gloss,
        chapterId,
        sentenceId: sentence.id,
        timestamp,
      });
      setDone(true);
      onSaved?.();
      setTimeout(onClose, 900);
    } finally {
      setSaving(false);
    }
  };

  const charChips = Array.from(sentence.zh.replace(/[，。！？、：；""''（）()\s]/g, "")).slice(0, 24);

  return (
    <div className="fixed inset-0 z-50 pb-safe">
      <div className="absolute inset-0 bg-ink/40 backdrop-blur-[2px]" onClick={onClose} aria-hidden />
      <div className="absolute inset-x-0 bottom-0 p-3 mx-auto max-w-md">
        <div className="ink-card p-5 animate-slide-up">
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2 text-ink-500">
              <BookmarkSolid className="text-sun" width={20} height={20} />
              <h3 className="font-medium">Save to vault</h3>
            </div>
            <button
              onClick={onClose}
              className="text-ink-300 hover:text-ink p-1 -m-1 transition"
              aria-label="Close"
            >
              <Close width={20} height={20} />
            </button>
          </div>

          <p className="font-zh text-lg leading-relaxed text-ink mb-3">{sentence.zh}</p>

          {sentence.idioms?.length ? (
            <div className="mb-3">
              <div className="text-xs uppercase tracking-wider text-ink-400 mb-1.5">Suggested</div>
              <div className="flex flex-wrap gap-1.5">
                {sentence.idioms.map((idi) => (
                  <button
                    key={idi.term}
                    onClick={() => setPicked({ term: idi.term, pinyin: idi.pinyin, meaning: idi.meaning })}
                    className={`chip transition ${
                      picked?.term === idi.term ? "bg-ink text-ink-50 border-ink" : "hover:bg-ink-100"
                    }`}
                  >
                    <span className="font-zh">{idi.term}</span>
                    <span className="opacity-70">· {idi.pinyin}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <div className="mb-4">
            <div className="text-xs uppercase tracking-wider text-ink-400 mb-1.5">Or tap a character</div>
            <div className="flex flex-wrap gap-1">
              {charChips.map((c, i) => (
                <button
                  key={`${c}-${i}`}
                  onClick={() => setPicked({ term: c })}
                  className={`w-9 h-9 rounded-md font-zh text-lg flex items-center justify-center transition ${
                    picked?.term === c
                      ? "bg-ink text-ink-50"
                      : "bg-ink-100 text-ink-500 hover:bg-ink-200"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {picked && !picked.meaning && (
            <input
              value={customMeaning}
              onChange={(e) => setCustomMeaning(e.target.value)}
              placeholder="Quick note (optional)"
              className="w-full mb-3 rounded-xl border border-ink-200 bg-white/80 px-4 py-2.5 text-sm placeholder:text-ink-300 focus:outline-none focus:border-ink"
            />
          )}

          <button
            onClick={handleSave}
            disabled={!picked || saving || done}
            className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {done ? (
              <>
                <Check width={18} height={18} /> Saved to vault
              </>
            ) : saving ? (
              "Saving…"
            ) : (
              <>
                <BookmarkSolid width={18} height={18} />
                Save{picked ? ` "${picked.term}"` : "…"}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
