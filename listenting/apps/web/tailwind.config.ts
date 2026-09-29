import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        serif: ["var(--font-serif)", "Georgia", "serif"],
        zh: ["var(--font-zh)", "system-ui", "sans-serif"],
      },
      colors: {
        ink: {
          DEFAULT: "#0F1115",
          50: "#F7F6F2",
          100: "#EDEAE2",
          200: "#D9D4C5",
          300: "#A8A192",
          400: "#6E6A5E",
          500: "#3D3A33",
          900: "#0F1115",
        },
        sun: {
          DEFAULT: "#D4452B",
          soft: "#E8826B",
        },
        jade: {
          DEFAULT: "#1F6B5E",
          soft: "#7BA89F",
        },
        gold: "#C8A24B",
      },
      animation: {
        "slide-up": "slide-up 0.45s cubic-bezier(0.22, 1, 0.36, 1)",
        "fade-in": "fade-in 0.5s ease-out",
        "pulse-soft": "pulse-soft 2.5s ease-in-out infinite",
        "wave": "wave 1.4s ease-in-out infinite",
        "ink-bleed": "ink-bleed 1.2s ease-out forwards",
      },
      keyframes: {
        "slide-up": {
          "0%": { transform: "translateY(100%)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.7", transform: "scale(1.02)" },
        },
        "wave": {
          "0%, 100%": { transform: "scaleY(0.35)" },
          "50%": { transform: "scaleY(1)" },
        },
        "ink-bleed": {
          "0%": { opacity: "0", filter: "blur(8px)" },
          "100%": { opacity: "1", filter: "blur(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
