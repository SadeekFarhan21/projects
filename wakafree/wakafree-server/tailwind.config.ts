import type { Config } from 'tailwindcss'

// "Slate & Mono" design system tokens
const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0c1117',
          bright: '#343a43',
        },
        container: {
          lowest: '#070a0f',
          low: '#12161d',
          DEFAULT: '#161b22',
          high: '#202630',
          highest: '#2a303a',
        },
        onsurface: {
          DEFAULT: '#e6e8ec',
          variant: '#b3b9c4',
        },
        outline: {
          DEFAULT: '#8b919d',
          variant: '#3d434e',
        },
        line: '#272d36',
        primary: {
          DEFAULT: '#b48ead',
          dim: '#9d7794',
          on: '#2b1f29',
        },
        secondary: {
          DEFAULT: '#c2c8d2',
          container: '#3a414d',
        },
        tertiary: {
          DEFAULT: '#fabd34',
        },
      },
      fontFamily: {
        sans: ['var(--font-geist)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-geist)', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

export default config
