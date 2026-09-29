export interface Sentence {
  /** Index within chapter */
  id: number;
  /** Original Chinese text (classical) */
  zh: string;
  /** Modern Chinese paraphrase from the source */
  gloss: string;
  /** English translation — populated by enrich-translations.ts (optional) */
  en?: string;
  /** Pinyin reading (optional, computed on demand) */
  pinyin?: string;
  /** Start time in seconds within the chapter mp3 (estimated if no audio yet) */
  start: number;
  /** End time in seconds */
  end: number;
  /** Idioms / cultural references in this sentence */
  idioms?: Idiom[];
}

export interface Idiom {
  /** The 4-character idiom or proper noun, exactly as written */
  term: string;
  /** Pinyin */
  pinyin: string;
  /** Literal character-by-character meaning */
  literal: string;
  /** Cultural / actual meaning */
  meaning: string;
  /** Optional story / origin */
  origin?: string;
  /** Category */
  kind: "chengyu" | "reference" | "concept";
}

export interface Chapter {
  /** "1" .. "13" */
  id: string;
  /** Original chapter title (Chinese) */
  titleZh: string;
  /** English title */
  titleEn: string;
  /** Pinyin title */
  titlePinyin: string;
  /** Total duration in seconds (estimated until mp3 is added) */
  duration: number;
  /** Audio source URL — may be empty if not yet recorded */
  audioSrc: string;
  /** Whether real audio exists; if false, app uses browser speechSynthesis */
  hasAudio: boolean;
  /** All sentences with timestamps */
  sentences: Sentence[];
}

export interface ChapterIndex {
  chapters: Array<{
    id: string;
    titleZh: string;
    titleEn: string;
    titlePinyin: string;
    duration: number;
    hasAudio: boolean;
    sentenceCount: number;
  }>;
}

export interface VaultEntry {
  id: string;            // uuid
  word: string;          // the saved word/phrase
  pinyin?: string;
  meaning: string;       // English meaning
  sentenceZh: string;    // full sentence context
  sentenceEn: string;
  chapterId: string;
  sentenceId: number;
  timestamp: number;     // audio timestamp at save
  savedAt: number;       // unix ms
  reviewCount: number;
  lastReviewedAt?: number;
}

export interface CheckinQuestion {
  question: string;
  contextSummary: string;
  /** Rubric the grader will check against */
  rubric: string;
}

export interface CheckinGrade {
  verdict: "great" | "good" | "partial" | "off";
  feedback: string;
  modelAnswer: string;
}
