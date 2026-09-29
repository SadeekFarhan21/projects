"use client";
import { useSettings } from "@/lib/settings";
import { hasBrowserSTT } from "@/lib/speech";
import { useEffect, useState } from "react";

export default function SettingsPage() {
  const [settings, update] = useSettings();
  const [sttOk, setSttOk] = useState<boolean | null>(null);
  useEffect(() => setSttOk(hasBrowserSTT()), []);

  return (
    <div className="mx-auto max-w-xl px-5 py-6">
      <header className="mb-6">
        <p className="text-xs uppercase tracking-widest text-ink-400 mb-1">Settings</p>
        <h1 className="font-serif-display text-3xl text-ink">How you listen</h1>
      </header>

      <section className="ink-card p-5 mb-4 space-y-5">
        <div>
          <label className="block text-sm font-medium text-ink mb-2">Voice check-ins</label>
          <p className="text-xs text-ink-400 mb-3">
            Every few minutes the narrator pauses and asks you a quick question about what just happened.
          </p>
          <div className="flex flex-wrap gap-2">
            {[0, 2, 3, 5, 8].map((n) => (
              <button
                key={n}
                onClick={() => update({ checkinIntervalMin: n })}
                className={`chip ${
                  settings.checkinIntervalMin === n
                    ? "bg-ink text-ink-50 border-ink"
                    : "hover:bg-ink-100"
                }`}
              >
                {n === 0 ? "Off" : `${n} min`}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-ink mb-2">Check-in language</label>
          <div className="flex gap-2">
            <button
              onClick={() => update({ questionLanguage: "en" })}
              className={`chip ${settings.questionLanguage === "en" ? "bg-ink text-ink-50 border-ink" : ""}`}
            >
              English
            </button>
            <button
              onClick={() => update({ questionLanguage: "zh" })}
              className={`chip ${settings.questionLanguage === "zh" ? "bg-ink text-ink-50 border-ink" : ""}`}
            >
              中文
            </button>
          </div>
        </div>
      </section>

      <section className="ink-card p-5 mb-4 space-y-4">
        <Toggle
          label="Show English translation"
          desc="Lazy-translated by Claude as you go, then cached."
          on={settings.showEnglish}
          onChange={(v) => update({ showEnglish: v })}
        />
        <Toggle
          label="Show modern Chinese paraphrase"
          desc="Side-by-side with the classical original."
          on={settings.showGloss}
          onChange={(v) => update({ showGloss: v })}
        />
        <Toggle
          label="Show pinyin"
          desc="Coming soon."
          on={settings.showPinyin}
          onChange={(v) => update({ showPinyin: v })}
          disabled
        />
      </section>

      <section className="ink-card p-5 text-sm text-ink-500 space-y-2">
        <div className="flex items-center justify-between">
          <span>Browser voice recognition</span>
          <span className={sttOk ? "text-jade" : "text-sun"}>
            {sttOk === null ? "…" : sttOk ? "Available" : "Not supported — we'll fall back to typed answers"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Service worker</span>
          <span className="text-ink-400">Installed — add to Home Screen for the mobile-app feel</span>
        </div>
      </section>
    </div>
  );
}

function Toggle({
  label,
  desc,
  on,
  onChange,
  disabled,
}: {
  label: string;
  desc?: string;
  on: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={() => !disabled && onChange(!on)}
      disabled={disabled}
      className="w-full flex items-center justify-between gap-4 text-left disabled:opacity-50"
    >
      <div>
        <div className="text-sm font-medium text-ink">{label}</div>
        {desc && <div className="text-xs text-ink-400 mt-0.5">{desc}</div>}
      </div>
      <span
        className={`relative inline-flex h-6 w-11 shrink-0 rounded-full transition ${
          on ? "bg-ink" : "bg-ink-200"
        }`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
            on ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </span>
    </button>
  );
}
