/**
 * Hand-curated idioms (chéngyǔ) and cultural references from Sun Tzu's Art of War.
 * Used by prepare-chapters.ts to flag occurrences in the transcript.
 *
 * Coverage is selective — only well-known, useful-to-learners idioms.
 * The pipeline matches by substring against the original Chinese text.
 */
import type { Idiom } from "../src/lib/types";

export const IDIOMS: Idiom[] = [
  {
    term: "知己知彼",
    pinyin: "zhī jǐ zhī bǐ",
    literal: "know-self know-other",
    meaning:
      "Know yourself and know your opponent — the foundation of every plan. Used today in business, sports, debate.",
    origin:
      "Famously paired with 百战不殆 — \"know yourself, know your enemy, in a hundred battles you will never be defeated.\"",
    kind: "chengyu",
  },
  {
    term: "百战不殆",
    pinyin: "bǎi zhàn bù dài",
    literal: "hundred battles not endangered",
    meaning: "Will never be defeated in a hundred battles. Modern usage: long-term resilience.",
    origin: "Pairs with 知己知彼 — the most-quoted line from this book.",
    kind: "chengyu",
  },
  {
    term: "百战百胜",
    pinyin: "bǎi zhàn bǎi shèng",
    literal: "hundred battles, hundred victories",
    meaning: "Invincible in every fight — but Sun Tzu calls this not the highest skill.",
    kind: "chengyu",
  },
  {
    term: "不战而屈人之兵",
    pinyin: "bù zhàn ér qū rén zhī bīng",
    literal: "without fighting subdue another's soldiers",
    meaning:
      "Defeat the enemy without a battle. The peak of strategy — winning by position, deception, or diplomacy.",
    kind: "concept",
  },
  {
    term: "兵不厌诈",
    pinyin: "bīng bù yàn zhà",
    literal: "soldiers don't tire of deception",
    meaning: "In war, deception is never excessive. A cornerstone proverb of strategy.",
    kind: "chengyu",
  },
  {
    term: "出其不意",
    pinyin: "chū qí bù yì",
    literal: "come-out-of (their) un-expected",
    meaning: "Catch them off guard. Still everyday Chinese — used about surprise moves, decisions, gifts.",
    kind: "chengyu",
  },
  {
    term: "攻其无备",
    pinyin: "gōng qí wú bèi",
    literal: "attack their without-preparation",
    meaning: "Strike where they are unprepared. Paired with 出其不意.",
    kind: "chengyu",
  },
  {
    term: "以逸待劳",
    pinyin: "yǐ yì dài láo",
    literal: "use rest to wait for (their) exhaustion",
    meaning: "Stay rested while the enemy tires themselves out. A patience-as-weapon concept.",
    kind: "chengyu",
  },
  {
    term: "围魏救赵",
    pinyin: "wéi wèi jiù zhào",
    literal: "besiege Wei to rescue Zhao",
    meaning: "Solve a problem by attacking an indirect target. From the Warring States — Sun Bin's stratagem.",
    origin: "One of the 36 stratagems; the idea is foreshadowed in Sun Tzu's emphasis on indirect approach.",
    kind: "reference",
  },
  {
    term: "声东击西",
    pinyin: "shēng dōng jī xī",
    literal: "make-noise east, strike west",
    meaning: "Feint in one direction, attack in another. A classic deception pattern.",
    kind: "chengyu",
  },
  {
    term: "兵贵神速",
    pinyin: "bīng guì shén sù",
    literal: "soldiers value divine speed",
    meaning: "In war, speed is everything. Modern usage: any time-critical operation.",
    kind: "chengyu",
  },
  {
    term: "兵贵胜",
    pinyin: "bīng guì shèng",
    literal: "soldiers value victory",
    meaning: "What matters in war is winning — not how long or how elegant.",
    kind: "chengyu",
  },
  {
    term: "上兵伐谋",
    pinyin: "shàng bīng fá móu",
    literal: "highest soldiering attacks (the enemy's) plans",
    meaning: "The best warfare attacks strategy itself — disrupt their plans before they form.",
    kind: "concept",
  },
  {
    term: "其疾如风",
    pinyin: "qí jí rú fēng",
    literal: "their swiftness like wind",
    meaning:
      "Swift as wind — first of the 风林火山 (wind/forest/fire/mountain) sequence, later borrowed by Japan's Takeda Shingen as his battle standard.",
    kind: "reference",
  },
  {
    term: "其徐如林",
    pinyin: "qí xú rú lín",
    literal: "their slowness like forest",
    meaning: "Silent as a forest. Second of the wind-forest-fire-mountain sequence.",
    kind: "reference",
  },
  {
    term: "侵掠如火",
    pinyin: "qīn lüè rú huǒ",
    literal: "invading-plundering like fire",
    meaning: "Devastating as fire. Third of the wind-forest-fire-mountain sequence.",
    kind: "reference",
  },
  {
    term: "不动如山",
    pinyin: "bù dòng rú shān",
    literal: "not-moving like mountain",
    meaning: "Immovable as a mountain. Fourth of the famous four. Used today for unshakeable resolve.",
    kind: "reference",
  },
  {
    term: "道",
    pinyin: "dào",
    literal: "the Way",
    meaning:
      "More than a road — the moral alignment between a ruler and the people. The first of Sun Tzu's five fundamentals.",
    kind: "concept",
  },
  {
    term: "天",
    pinyin: "tiān",
    literal: "Heaven",
    meaning: "Weather, season, timing — the second fundamental. Conditions beyond your control.",
    kind: "concept",
  },
  {
    term: "地",
    pinyin: "dì",
    literal: "Earth / Ground",
    meaning: "Terrain — distance, danger, openness. The third fundamental.",
    kind: "concept",
  },
  {
    term: "将",
    pinyin: "jiàng",
    literal: "General",
    meaning: "Leadership — wisdom, trust, benevolence, courage, discipline. The fourth fundamental.",
    kind: "concept",
  },
  {
    term: "法",
    pinyin: "fǎ",
    literal: "Method / Law",
    meaning: "Organization — chain of command, logistics, regulation. The fifth fundamental.",
    kind: "concept",
  },
  {
    term: "势",
    pinyin: "shì",
    literal: "Strategic configuration",
    meaning:
      "One of the hardest concepts to translate — the latent force in a situation, the way water gains power from a high cliff. Used in business, politics, martial arts.",
    kind: "concept",
  },
  {
    term: "诡道",
    pinyin: "guǐ dào",
    literal: "deceptive way",
    meaning:
      "\"War is the way of deception.\" One of Sun Tzu's most quoted lines: 兵者，诡道也.",
    kind: "concept",
  },
  {
    term: "速战速决",
    pinyin: "sù zhàn sù jué",
    literal: "fast battle, fast decision",
    meaning: "Win fast. Don't drag wars on. Now used about any project that should be wrapped quickly.",
    kind: "chengyu",
  },
  {
    term: "千里馈粮",
    pinyin: "qiān lǐ kuì liáng",
    literal: "thousand li delivering grain",
    meaning:
      "Shipping food a thousand li (≈500km) to feed an army — Sun Tzu's vivid image for the cost of long campaigns.",
    kind: "reference",
  },
  {
    term: "日费千金",
    pinyin: "rì fèi qiān jīn",
    literal: "daily expense (of) a thousand pieces of gold",
    meaning: "A vast daily cost. Still used today for any expensive operation.",
    kind: "chengyu",
  },
  {
    term: "孙子曰",
    pinyin: "Sūn zǐ yuē",
    literal: "Sun Tzu says",
    meaning:
      "\"Sun Tzu says.\" The traditional opening line of each chapter, like \"Confucius said\" (子曰) in the Analects.",
    kind: "reference",
  },
];

/** Find idioms present in a given Chinese sentence. */
export function findIdiomsIn(zh: string): Idiom[] {
  const seen = new Set<string>();
  const out: Idiom[] = [];
  for (const idiom of IDIOMS) {
    if (zh.includes(idiom.term) && !seen.has(idiom.term)) {
      seen.add(idiom.term);
      out.push(idiom);
    }
  }
  return out;
}
