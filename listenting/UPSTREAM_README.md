# ListenTing

ListenTing is a monorepo for the hackathon project: a language-learning web app plus a slide deck used to present it.

## Workspace

- `apps/web`: Next.js app
- `slides`: Astro slide deck

## Getting Started

```bash
git clone git@github.com:SadeekFarhan21/ListenTing.git
cd Anthropic-Hackathon
pnpm install
cp apps/web/.env.local.example apps/web/.env.local
```

Start the web app from the repo root:

```bash
pnpm dev
```

This runs the Next.js app in `apps/web` on `http://localhost:3000`.

## Environment Variables

Set secrets in `apps/web/.env.local`. Do not commit real keys.

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=optional_voice_id
ELEVENLABS_VOICE_ID_ZH=optional_voice_id_for_chinese
ELEVENLABS_MODEL_ID=optional_model_id
```

## Useful Commands

```bash
pnpm dev
pnpm build
pnpm --filter web lint
pnpm --filter web prepare-chapters
pnpm --filter web enrich-translations
pnpm --filter web generate-audio
```
