"use client";

interface Props {
  /** When true, bars dance. When false, they idle low. */
  active?: boolean;
  count?: number;
  className?: string;
}

export function Waveform({ active = true, count = 5, className = "" }: Props) {
  return (
    <span className={`inline-flex items-end h-6 ${className}`} aria-hidden="true">
      {Array.from({ length: count }).map((_, i) => (
        <span
          key={i}
          className="wave-bar"
          style={{
            height: `${100 - Math.abs(i - (count - 1) / 2) * 18}%`,
            animationName: active ? "wave" : "none",
            animationDuration: "1.2s",
            animationTimingFunction: "ease-in-out",
            animationIterationCount: "infinite",
            animationDelay: `${i * 90}ms`,
            opacity: active ? 1 : 0.35,
          }}
        />
      ))}
    </span>
  );
}
