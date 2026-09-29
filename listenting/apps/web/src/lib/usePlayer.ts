"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Chapter, Sentence } from "./types";

interface PlayerState {
  isPlaying: boolean;
  isReady: boolean;
  currentTime: number;
  duration: number;
  activeSentenceId: number | null;
  /** "audio" when an mp3 is loaded, "tts" when falling back to browser SpeechSynthesis */
  source: "audio" | "tts" | "idle";
  error: string | null;
}

interface UsePlayerOptions {
  rate?: number;
  onSentenceEnter?: (s: Sentence) => void;
  onSentenceLeave?: (s: Sentence) => void;
}

/**
 * Audio engine that prefers a real chapter mp3 (apps/web/public/audio/chN.mp3)
 * and falls back to the browser SpeechSynthesis API when the file is missing.
 *
 * In TTS mode the engine plays sentence-by-sentence using `SpeechSynthesisUtterance`,
 * advancing a virtual clock so the transcript-highlight and check-in scheduler work
 * the same way as with real audio.
 */
export function usePlayer(chapter: Chapter | null, opts: UsePlayerOptions = {}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const ttsTickRef = useRef<number | null>(null);
  const ttsStartedAtRef = useRef<number>(0);
  const ttsBaseTimeRef = useRef<number>(0);
  const ttsUtterRef = useRef<SpeechSynthesisUtterance | null>(null);
  const rateRef = useRef<number>(opts.rate ?? 1);
  const prevActiveRef = useRef<number | null>(null);

  const [state, setState] = useState<PlayerState>({
    isPlaying: false,
    isReady: false,
    currentTime: 0,
    duration: chapter?.duration ?? 0,
    activeSentenceId: null,
    source: "idle",
    error: null,
  });

  // Resolve which mp3 (if any) is available for this chapter
  useEffect(() => {
    if (!chapter) return;
    let cancelled = false;
    setState((s) => ({ ...s, isReady: false, currentTime: 0, source: "idle", duration: chapter.duration, error: null }));
    const tryAudio = async () => {
      try {
        const head = await fetch(chapter.audioSrc, { method: "HEAD" });
        if (cancelled) return;
        if (head.ok && head.headers.get("content-type")?.includes("audio")) {
          // Real mp3 available
          if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.src = "";
          }
          const a = new Audio(chapter.audioSrc);
          a.preload = "metadata";
          a.playbackRate = rateRef.current;
          a.addEventListener("loadedmetadata", () => {
            if (cancelled) return;
            setState((s) => ({
              ...s,
              isReady: true,
              duration: Number.isFinite(a.duration) ? a.duration : chapter.duration,
              source: "audio",
            }));
          });
          a.addEventListener("ended", () => {
            setState((s) => ({ ...s, isPlaying: false }));
          });
          a.addEventListener("error", () => {
            // Fall through to TTS
            setupTts();
          });
          audioRef.current = a;
        } else {
          setupTts();
        }
      } catch {
        setupTts();
      }
    };
    const setupTts = () => {
      if (cancelled) return;
      audioRef.current = null;
      setState((s) => ({ ...s, isReady: true, source: "tts", duration: chapter.duration }));
    };
    tryAudio();
    return () => {
      cancelled = true;
      stopAll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chapter?.id]);

  const stopAll = useCallback(() => {
    if (audioRef.current) audioRef.current.pause();
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    if (ttsTickRef.current != null) {
      cancelAnimationFrame(ttsTickRef.current);
      ttsTickRef.current = null;
    }
  }, []);

  // Compute active sentence on time change
  const sentences = chapter?.sentences ?? [];
  const findActive = useCallback(
    (t: number): Sentence | null => {
      if (!sentences.length) return null;
      // binary search
      let lo = 0,
        hi = sentences.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const s = sentences[mid];
        if (t < s.start) hi = mid - 1;
        else if (t > s.end) lo = mid + 1;
        else return s;
      }
      // If between sentences, return the most recent one
      const idx = Math.max(0, lo - 1);
      return sentences[idx] ?? null;
    },
    [sentences],
  );

  const updateActive = useCallback(
    (t: number) => {
      const active = findActive(t);
      const newId = active?.id ?? null;
      if (newId !== prevActiveRef.current) {
        const prev = prevActiveRef.current;
        prevActiveRef.current = newId;
        if (prev != null) {
          const prevS = sentences.find((x) => x.id === prev);
          if (prevS) opts.onSentenceLeave?.(prevS);
        }
        if (active) opts.onSentenceEnter?.(active);
        setState((s) => ({ ...s, activeSentenceId: newId }));
      }
    },
    [findActive, opts, sentences],
  );

  // --- mp3 mode: rAF loop driving currentTime ---
  useEffect(() => {
    if (state.source !== "audio" || !audioRef.current) return;
    let raf = 0;
    const a = audioRef.current;
    const tick = () => {
      const t = a.currentTime;
      setState((s) => (s.currentTime !== t ? { ...s, currentTime: t } : s));
      updateActive(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [state.source, updateActive]);

  // --- tts mode: virtual clock driven by performance.now ---
  const startTtsClock = useCallback(() => {
    if (ttsTickRef.current != null) cancelAnimationFrame(ttsTickRef.current);
    ttsStartedAtRef.current = performance.now();
    const tick = () => {
      const elapsed = (performance.now() - ttsStartedAtRef.current) / 1000;
      const t = ttsBaseTimeRef.current + elapsed * rateRef.current;
      setState((s) => (s.currentTime !== t ? { ...s, currentTime: t } : s));
      updateActive(t);
      if (chapter && t >= chapter.duration) {
        stopAll();
        setState((s) => ({ ...s, isPlaying: false, currentTime: chapter.duration }));
        return;
      }
      ttsTickRef.current = requestAnimationFrame(tick);
    };
    ttsTickRef.current = requestAnimationFrame(tick);
  }, [chapter, updateActive, stopAll]);

  const speakFrom = useCallback(
    (startTime: number) => {
      if (!chapter || state.source !== "tts") return;
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      const synth = window.speechSynthesis;
      synth.cancel();
      const remaining = chapter.sentences.filter((s) => s.end > startTime);
      if (remaining.length === 0) return;

      const speakNext = (idx: number) => {
        if (idx >= remaining.length) return;
        const s = remaining[idx];
        const u = new SpeechSynthesisUtterance(s.zh);
        u.lang = "zh-CN";
        u.rate = rateRef.current;
        u.onend = () => speakNext(idx + 1);
        ttsUtterRef.current = u;
        synth.speak(u);
      };
      speakNext(0);
    },
    [chapter, state.source],
  );

  const play = useCallback(() => {
    if (!chapter) return;
    if (state.source === "audio" && audioRef.current) {
      audioRef.current.playbackRate = rateRef.current;
      audioRef.current.play();
      setState((s) => ({ ...s, isPlaying: true }));
    } else if (state.source === "tts") {
      ttsBaseTimeRef.current = state.currentTime;
      startTtsClock();
      speakFrom(state.currentTime);
      setState((s) => ({ ...s, isPlaying: true }));
    }
  }, [chapter, state.source, state.currentTime, startTtsClock, speakFrom]);

  const pause = useCallback(() => {
    if (state.source === "audio" && audioRef.current) {
      audioRef.current.pause();
    } else if (state.source === "tts") {
      if (ttsTickRef.current != null) {
        cancelAnimationFrame(ttsTickRef.current);
        ttsTickRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    }
    setState((s) => ({ ...s, isPlaying: false }));
  }, [state.source]);

  const seek = useCallback(
    (t: number) => {
      const clamped = Math.max(0, Math.min(t, state.duration || 0));
      if (state.source === "audio" && audioRef.current) {
        audioRef.current.currentTime = clamped;
        setState((s) => ({ ...s, currentTime: clamped }));
        updateActive(clamped);
      } else if (state.source === "tts") {
        const wasPlaying = state.isPlaying;
        if (typeof window !== "undefined" && "speechSynthesis" in window) {
          window.speechSynthesis.cancel();
        }
        if (ttsTickRef.current != null) {
          cancelAnimationFrame(ttsTickRef.current);
          ttsTickRef.current = null;
        }
        ttsBaseTimeRef.current = clamped;
        setState((s) => ({ ...s, currentTime: clamped }));
        updateActive(clamped);
        if (wasPlaying) {
          startTtsClock();
          speakFrom(clamped);
        }
      }
    },
    [state.source, state.duration, state.isPlaying, updateActive, startTtsClock, speakFrom],
  );

  const setRate = useCallback(
    (r: number) => {
      rateRef.current = r;
      if (audioRef.current) audioRef.current.playbackRate = r;
      // Restart TTS at new rate for it to take effect on subsequent utterances
      if (state.source === "tts" && state.isPlaying) {
        if (typeof window !== "undefined" && "speechSynthesis" in window) {
          window.speechSynthesis.cancel();
        }
        ttsBaseTimeRef.current = state.currentTime;
        startTtsClock();
        speakFrom(state.currentTime);
      }
    },
    [state.source, state.isPlaying, state.currentTime, startTtsClock, speakFrom],
  );

  const activeSentence = useMemo(
    () => (state.activeSentenceId == null ? null : sentences.find((s) => s.id === state.activeSentenceId) ?? null),
    [state.activeSentenceId, sentences],
  );

  useEffect(() => () => stopAll(), [stopAll]);

  return {
    state,
    activeSentence,
    play,
    pause,
    seek,
    setRate,
    stop: stopAll,
  };
}
