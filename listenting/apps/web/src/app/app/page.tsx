import Link from "next/link";
import { getIndex } from "@/lib/data";
import { fmtTime } from "@/lib/cn";
import { Play, Sparkle } from "@/components/Icons";

export const dynamic = "force-dynamic";

export default async function AppHome() {
  const index = await getIndex();
  const totalMin = Math.round(index.chapters.reduce((n, c) => n + c.duration, 0) / 60);
  const audioCount = index.chapters.filter((c) => c.hasAudio).length;

  return (
    <div className="mx-auto max-w-xl px-5 py-6">
      <header className="mb-6">
        <p className="text-xs uppercase tracking-widest text-ink-400 mb-1">Tīng · 听</p>
        <h1 className="font-serif-display text-3xl text-ink leading-tight">
          The Art of War
        </h1>
        <p className="font-zh text-lg text-ink-500 mt-1">孙子兵法 · Sūn Zǐ Bīng Fǎ</p>
        <p className="text-sm text-ink-400 mt-3">
          {index.chapters.length} chapters · ~{totalMin} min · {audioCount} with recorded audio
        </p>
      </header>

      {audioCount === 0 && (
        <div className="ink-card p-4 mb-6 flex items-start gap-3">
          <Sparkle className="text-gold mt-0.5 shrink-0" width={18} height={18} />
          <div className="text-sm text-ink-500 leading-relaxed">
            No chapter MP3s yet — try a chapter and we'll narrate it with your device's voice in the meantime.
            Generate the real audio anytime with:
            <code className="block mt-2 font-mono text-xs bg-ink-100 px-2 py-1.5 rounded">
              ELEVENLABS_API_KEY=… pnpm generate-audio
            </code>
          </div>
        </div>
      )}

      <ul className="space-y-2">
        {index.chapters.map((c) => (
          <li key={c.id}>
            <Link
              href={`/app/listen/${c.id}`}
              className="ink-card flex items-center gap-4 p-4 hover:translate-y-[-1px] transition will-change-transform"
            >
              <div className="w-12 h-12 rounded-full bg-ink text-ink-50 flex items-center justify-center font-serif-display text-lg shrink-0">
                {c.id}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-serif-display text-ink text-lg truncate">
                  <span className="font-zh">{c.titleZh}</span>
                  <span className="text-ink-300"> · </span>
                  <span>{c.titleEn}</span>
                </div>
                <div className="text-xs text-ink-400 mt-0.5">
                  {c.titlePinyin} · {fmtTime(c.duration)} · {c.sentenceCount} lines
                  {c.hasAudio ? "" : " · synth voice"}
                </div>
              </div>
              <Play className="text-ink-300" width={22} height={22} />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
