"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import type { VaultEntry } from "@/lib/types";
import { deleteEntry, listEntries, markReviewed } from "@/lib/vault";
import { Layers, Play, Trash, Check } from "@/components/Icons";
import { speak } from "@/lib/speech";

export default function VaultPage() {
  const [entries, setEntries] = useState<VaultEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    const list = await listEntries();
    setEntries(list);
    setLoading(false);
  };
  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="mx-auto max-w-xl px-5 py-6">
      <header className="mb-6">
        <p className="text-xs uppercase tracking-widest text-ink-400 mb-1">Vault · 字库</p>
        <h1 className="font-serif-display text-3xl text-ink">Your saved words</h1>
        <p className="text-sm text-ink-400 mt-2">
          {entries.length === 0
            ? "Nothing saved yet — tap the bookmark while listening to a chapter."
            : `${entries.length} entr${entries.length === 1 ? "y" : "ies"} · tap to hear them again`}
        </p>
      </header>

      {loading ? (
        <p className="text-ink-400 text-sm">Loading…</p>
      ) : entries.length === 0 ? (
        <div className="ink-card p-6 text-center">
          <Layers className="mx-auto text-ink-300 mb-3" width={28} height={28} />
          <p className="text-ink-500 mb-4">Saved words live here.</p>
          <Link href="/app" className="btn-primary">
            <Play width={16} height={16} />
            Start a chapter
          </Link>
        </div>
      ) : (
        <ul className="space-y-3">
          {entries.map((e) => (
            <li key={e.id} className="ink-card p-4">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="min-w-0">
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="font-zh text-2xl text-ink">{e.word}</span>
                    {e.pinyin && <span className="text-sm text-gold">{e.pinyin}</span>}
                  </div>
                  <p className="text-sm text-ink-500">{e.meaning}</p>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    onClick={() => speak(e.word, "zh")}
                    className="p-2 text-ink-300 hover:text-ink"
                    aria-label="Speak"
                  >
                    <Play width={18} height={18} />
                  </button>
                  <button
                    onClick={async () => {
                      await markReviewed(e.id);
                      await refresh();
                    }}
                    className="p-2 text-ink-300 hover:text-jade"
                    aria-label="Mark reviewed"
                  >
                    <Check width={18} height={18} />
                  </button>
                  <button
                    onClick={async () => {
                      await deleteEntry(e.id);
                      await refresh();
                    }}
                    className="p-2 text-ink-300 hover:text-sun"
                    aria-label="Delete"
                  >
                    <Trash width={18} height={18} />
                  </button>
                </div>
              </div>
              <div className="text-xs text-ink-400 mt-3 border-t border-ink-200/50 pt-3">
                <p className="font-zh text-ink-500 leading-relaxed">{e.sentenceZh}</p>
                {e.sentenceEn && <p className="italic mt-1">{e.sentenceEn}</p>}
                <div className="flex items-center justify-between mt-2">
                  <Link
                    href={`/app/listen/${e.chapterId}`}
                    className="text-ink-400 hover:text-ink underline decoration-dotted underline-offset-4"
                  >
                    Chapter {e.chapterId} · {Math.floor(e.timestamp / 60)}:
                    {Math.floor(e.timestamp % 60)
                      .toString()
                      .padStart(2, "0")}
                  </Link>
                  <span>
                    Reviewed {e.reviewCount}×
                  </span>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
