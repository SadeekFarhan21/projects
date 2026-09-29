"use client";
import { useEffect, useState } from "react";

export interface Settings {
  checkinIntervalMin: number; // minutes between voice check-ins (0 = off)
  questionLanguage: "en" | "zh";
  showPinyin: boolean;
  showEnglish: boolean;
  showGloss: boolean;
  playbackRate: number;
}

const DEFAULTS: Settings = {
  checkinIntervalMin: 3,
  questionLanguage: "en",
  showPinyin: false,
  showEnglish: true,
  showGloss: false,
  playbackRate: 1,
};

const KEY = "ting.settings.v1";

export function loadSettings(): Settings {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}

export function saveSettings(s: Settings) {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, JSON.stringify(s));
}

export function useSettings(): [Settings, (next: Partial<Settings>) => void] {
  const [state, setState] = useState<Settings>(DEFAULTS);
  useEffect(() => {
    setState(loadSettings());
  }, []);
  const update = (next: Partial<Settings>) => {
    setState((s) => {
      const merged = { ...s, ...next };
      saveSettings(merged);
      return merged;
    });
  };
  return [state, update];
}
