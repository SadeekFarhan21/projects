import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;

const base = (p: P) => ({
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...p,
});

export const Play = (p: P) => (
  <svg {...base(p)}>
    <path d="M7 5.5v13a1 1 0 0 0 1.55.83l10-6.5a1 1 0 0 0 0-1.66l-10-6.5A1 1 0 0 0 7 5.5z" fill="currentColor" stroke="none" />
  </svg>
);
export const Pause = (p: P) => (
  <svg {...base(p)}>
    <rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" stroke="none" />
    <rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" stroke="none" />
  </svg>
);
export const Back15 = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 5a7 7 0 1 0 6.93 8" />
    <path d="M12 5 8 9l4 4" />
    <text x="12" y="16" textAnchor="middle" fontSize="6" fill="currentColor" stroke="none" fontFamily="ui-sans-serif">15</text>
  </svg>
);
export const Fwd30 = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 5a7 7 0 1 1-6.93 8" />
    <path d="m12 5 4 4-4 4" />
    <text x="12" y="16" textAnchor="middle" fontSize="6" fill="currentColor" stroke="none" fontFamily="ui-sans-serif">30</text>
  </svg>
);
export const Mic = (p: P) => (
  <svg {...base(p)}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0" />
    <path d="M12 18v3" />
  </svg>
);
export const Bookmark = (p: P) => (
  <svg {...base(p)}>
    <path d="M6 4h12v17l-6-4-6 4V4z" />
  </svg>
);
export const BookmarkSolid = (p: P) => (
  <svg {...base(p)}>
    <path d="M6 4h12v17l-6-4-6 4V4z" fill="currentColor" />
  </svg>
);
export const Sparkle = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.5 5.5l2 2M16.5 16.5l2 2M5.5 18.5l2-2M16.5 7.5l2-2" />
    <circle cx="12" cy="12" r="2.5" fill="currentColor" stroke="none" />
  </svg>
);
export const Book = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 4h7a4 4 0 0 1 4 4v12a3 3 0 0 0-3-3H4z" />
    <path d="M20 4h-7a4 4 0 0 0-4 4v12a3 3 0 0 1 3-3h8z" />
  </svg>
);
export const Layers = (p: P) => (
  <svg {...base(p)}>
    <path d="m12 3 9 5-9 5-9-5 9-5z" />
    <path d="m3 12 9 5 9-5" />
    <path d="m3 17 9 5 9-5" />
  </svg>
);
export const Gear = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);
export const Close = (p: P) => (
  <svg {...base(p)}>
    <path d="m6 6 12 12M18 6 6 18" />
  </svg>
);
export const ArrowLeft = (p: P) => (
  <svg {...base(p)}>
    <path d="M19 12H5" />
    <path d="m12 19-7-7 7-7" />
  </svg>
);
export const ArrowRight = (p: P) => (
  <svg {...base(p)}>
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);
export const Check = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 12.5 10 18.5 20 6.5" />
  </svg>
);
export const Volume = (p: P) => (
  <svg {...base(p)}>
    <path d="M11 5 6 9H3v6h3l5 4z" />
    <path d="M15 9a4 4 0 0 1 0 6" />
    <path d="M18 6a8 8 0 0 1 0 12" />
  </svg>
);
export const Trash = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14" />
  </svg>
);
