# Client-Side Audio Transcription Tool

Privacy-first audio transcription built with Next.js, Web Workers, and `@xenova/transformers`.
All processing runs in the browser (WASM + WebGPU/WebAssembly backend), so audio files are not uploaded to a server.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Web Worker for off-main-thread inference
- `@xenova/transformers` with `Xenova/whisper-base`

## Local Development

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

Production checks:

```bash
pnpm build
pnpm start
```

## Push to GitHub

If this is a new repository:

```bash
git init
git add .
git commit -m "feat: client-side audio transcription app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the repository already exists remotely:

```bash
git add .
git commit -m "feat: step 5 vercel build config and smart export"
git push
```

## Deploy on Vercel

1. Push your latest code to GitHub.
2. Go to Vercel: `https://vercel.com/new`
3. Import the GitHub repository.
4. Keep default framework detection (`Next.js`).
5. No environment variables are required for this app.
6. Click **Deploy**.

After deploy:

- First transcription is slower because Whisper model files download and cache in the browser.
- Later transcriptions are faster due browser cache.
- `whisper-base` is more stable against hallucination/repetition but has a larger first download and slower inference than `whisper-tiny`.

## Environment Variables

No environment variables are required for this client-side application.

## Troubleshooting

- If deployment fails on WASM/module resolution, confirm `next.config.mjs` exists and includes:
  - WASM asset rule (`test: /\.wasm$/`)
  - `asyncWebAssembly` experiment
  - SSR-safe aliases for `sharp` and `onnxruntime-node`
- If transcription does not start:
  - Use a modern Chromium/Firefox browser with Web Workers, WebAssembly, and Web Audio enabled.
  - Keep the tab active during long lecture processing.
