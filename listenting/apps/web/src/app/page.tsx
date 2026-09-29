import Link from "next/link";
import { ArrowRight, Bookmark, Mic, Sparkle } from "@/components/Icons";
import { Waveform } from "@/components/Waveform";

export default function Landing() {
  return (
    <main className="min-h-dvh bg-ink-50 text-ink overflow-hidden">
      {/* Top nav */}
      <header className="absolute inset-x-0 top-0 z-20">
        <div className="mx-auto max-w-6xl px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-ink text-ink-50 flex items-center justify-center font-zh text-base shadow-sm">
              听
            </div>
            <div className="font-serif-display text-lg">Tīng</div>
          </div>
          <nav className="flex items-center gap-2">
            <Link href="#features" className="hidden sm:inline btn-ghost">
              Features
            </Link>
            <Link href="#how" className="hidden sm:inline btn-ghost">
              How it works
            </Link>
            <Link href="/app" className="btn-primary text-sm">
              Open app
              <ArrowRight width={16} height={16} />
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative pt-32 sm:pt-40 pb-24">
        <div className="absolute inset-0 -z-10 opacity-40 paper" />
        <div className="mx-auto max-w-6xl px-6">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="chip mb-5">
                <Sparkle className="text-gold" width={12} height={12} />
                Mobile audiobook · Claude + ElevenLabs
              </div>
              <h1 className="font-serif-display text-[clamp(2.5rem,6vw,4.5rem)] leading-[1.05] tracking-tight">
                Listen your way <br />
                <span className="text-sun">into Chinese.</span>
              </h1>
              <p className="mt-6 text-lg text-ink-500 max-w-lg leading-relaxed">
                Tīng turns your commute into a Chinese-learning loop: classical audiobooks paired with a friendly
                AI that quizzes you, explains idioms, and remembers every word you save.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <Link href="/app" className="btn-sun text-base px-6 py-3.5">
                  Start listening
                  <ArrowRight width={18} height={18} />
                </Link>
                <Link href="#features" className="btn-ghost text-base">
                  See how it works
                </Link>
              </div>
              <p className="mt-6 text-xs text-ink-400">
                Today's library: <span className="font-zh">孙子兵法</span> · The Art of War — 13 chapters
              </p>
            </div>

            {/* Phone mockup */}
            <div className="relative mx-auto w-full max-w-sm">
              <div className="relative aspect-[9/19] rounded-[3rem] bg-ink shadow-2xl shadow-ink/40 p-3 border border-ink-900">
                <div className="absolute top-3 left-1/2 -translate-x-1/2 h-6 w-32 bg-ink-900 rounded-b-2xl z-10" />
                <div className="w-full h-full rounded-[2.4rem] bg-ink-50 overflow-hidden relative">
                  <div className="paper absolute inset-0 opacity-60" />
                  <div className="relative p-5 pt-10">
                    <div className="text-xs uppercase tracking-widest text-ink-400">Chapter 3 · Móu Gōng</div>
                    <div className="font-serif-display text-xl mt-1 mb-3">
                      <span className="font-zh">谋攻</span> · Attack by Strategy
                    </div>
                    <p className="font-zh text-2xl text-ink leading-snug mb-2">
                      故曰：
                      <span className="text-sun underline decoration-sun/40 decoration-2 underline-offset-4">
                        知己知彼
                      </span>
                      ，
                      <span className="text-sun underline decoration-sun/40 decoration-2 underline-offset-4">
                        百战不殆
                      </span>
                      。
                    </p>
                    <p className="italic text-sm text-ink-500 leading-relaxed">
                      "Know yourself and know your enemy — in a hundred battles, you will never be in peril."
                    </p>

                    <div className="mt-6 ink-card-dark p-3">
                      <div className="text-[10px] uppercase tracking-widest text-gold mb-1">Chéngyǔ · 成语</div>
                      <div className="font-zh text-2xl text-white">知己知彼</div>
                      <div className="text-xs text-ink-200 mt-1">
                        zhī jǐ zhī bǐ — know-self know-other
                      </div>
                    </div>

                    <div className="absolute bottom-5 inset-x-5">
                      <div className="ink-card-dark px-3 py-2.5 flex items-center justify-between">
                        <Waveform active className="text-gold" />
                        <div className="bg-sun text-white w-12 h-12 rounded-full flex items-center justify-center">
                          <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
                            <path d="M7 5.5v13a1 1 0 0 0 1.55.83l10-6.5a1 1 0 0 0 0-1.66l-10-6.5A1 1 0 0 0 7 5.5z" />
                          </svg>
                        </div>
                        <div className="flex items-center gap-1 text-ink-100 text-xs">
                          <Bookmark width={16} height={16} />
                          <Mic width={16} height={16} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating callouts */}
              <div className="hidden md:block absolute -left-8 top-1/3 ink-card px-4 py-3 max-w-[200px] animate-fade-in">
                <div className="text-[10px] uppercase tracking-widest text-gold mb-1">Cultural pop-up</div>
                <div className="text-sm">Cards explain the literal vs. cultural meaning the moment they're spoken.</div>
              </div>
              <div className="hidden md:block absolute -right-6 bottom-1/4 ink-card px-4 py-3 max-w-[200px] animate-fade-in">
                <div className="text-[10px] uppercase tracking-widest text-jade mb-1">Voice check-in</div>
                <div className="text-sm">"Why did the general avoid attacking the city?" — answer out loud.</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24 bg-white/60 border-y border-ink-200/50">
        <div className="mx-auto max-w-6xl px-6">
          <div className="max-w-2xl mb-12">
            <p className="text-xs uppercase tracking-widest text-ink-400 mb-2">Three things, done well</p>
            <h2 className="font-serif-display text-4xl leading-tight">
              The audiobook that <span className="text-sun">talks back.</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              accent="sun"
              icon={<Mic width={20} height={20} />}
              eyebrow="01 · Voice check-ins"
              title="Pause, ask, listen."
              body="Every few minutes the narration fades and Claude asks a comprehension question. Speak your answer in English or Chinese — the model grades you and resumes the book."
            />
            <FeatureCard
              accent="gold"
              icon={<Sparkle width={20} height={20} />}
              eyebrow="02 · Cultural pop-ups"
              title="Idioms in their world."
              body="Chéngyǔ and references trigger a card at the exact moment they're spoken. See the literal meaning, the cultural meaning, and the story behind it."
            />
            <FeatureCard
              accent="jade"
              icon={<Bookmark width={20} height={20} />}
              eyebrow="03 · Personal vocab vault"
              title="Save what sticks."
              body="Tap once to save the current sentence. Each entry becomes a smart flashcard with the audio snippet, pinyin, and meaning — synced to your device."
            />
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="py-24">
        <div className="mx-auto max-w-5xl px-6">
          <div className="max-w-2xl mb-10">
            <p className="text-xs uppercase tracking-widest text-ink-400 mb-2">Under the hood</p>
            <h2 className="font-serif-display text-4xl leading-tight">A small graph of moving parts.</h2>
          </div>

          <div className="ink-card p-6 sm:p-10 overflow-x-auto">
            <pre className="text-xs sm:text-sm leading-relaxed font-mono text-ink-500 whitespace-pre">
{`[Audiobook MP3]    ────►  Time-synced transcript (Chinese · pinyin · English)
        │
        ├── timestamp events ──►  Cultural pop-ups (chéngyǔ + references)
        │
        ├── smart pauses ─────►  Claude generates question → ElevenLabs reads it
        │                              │
        │                              └► Web Speech API listens → Claude grades
        │
        └── tap "save" ───────►  Vault entry stored locally (audio · pinyin · meaning)`}
            </pre>
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="pb-24">
        <div className="mx-auto max-w-3xl px-6 text-center">
          <h2 className="font-serif-display text-3xl sm:text-4xl mb-4">
            Open it on your phone. <span className="text-sun">Add to Home Screen.</span>
          </h2>
          <p className="text-ink-500 mb-6 max-w-xl mx-auto">
            Tīng is a PWA — it installs like a native app, runs offline once you've opened a chapter, and uses your
            phone's mic for the check-ins.
          </p>
          <Link href="/app" className="btn-sun text-base px-7 py-4">
            Open the app
            <ArrowRight width={18} height={18} />
          </Link>
          <p className="text-xs text-ink-400 mt-6">
            <span className="font-zh">听</span> · Tīng — built for the Anthropic Cultural Exchange hackathon ·
            Powered by Claude & ElevenLabs
          </p>
        </div>
      </section>
    </main>
  );
}

function FeatureCard({
  accent,
  icon,
  eyebrow,
  title,
  body,
}: {
  accent: "sun" | "gold" | "jade";
  icon: React.ReactNode;
  eyebrow: string;
  title: string;
  body: string;
}) {
  const accentClass =
    accent === "sun" ? "text-sun bg-sun/10" : accent === "gold" ? "text-gold bg-gold/10" : "text-jade bg-jade/10";
  return (
    <article className="ink-card p-6">
      <div className={`inline-flex items-center justify-center w-10 h-10 rounded-xl ${accentClass} mb-4`}>
        {icon}
      </div>
      <div className="text-xs uppercase tracking-widest text-ink-400 mb-1">{eyebrow}</div>
      <h3 className="font-serif-display text-2xl mb-2">{title}</h3>
      <p className="text-sm text-ink-500 leading-relaxed">{body}</p>
    </article>
  );
}
