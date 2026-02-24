# Audio Transcription Tool

Privacy-first transcription app built with Next.js.

- Desktop: runs locally in the browser with Whisper Small (`@huggingface/transformers`) via WebGPU/WASM.
- Mobile: uses secure Groq cloud fallback (`whisper-large-v3`) for stability and speed.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Web Worker inference pipeline
- `@huggingface/transformers`

## Local development

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

Production check:

```bash
pnpm build
pnpm start
```

## Environment variables

Create `.env.local`:

```bash
GROQ_API_KEY=your_groq_api_key
```

Notes:

- `GROQ_API_KEY` is required for mobile cloud fallback (`/api/transcribe`).
- Desktop local transcription works without cloud calls.

## Deploy (Vercel)

1. Push this repo to GitHub.
2. Import the repo in Vercel.
3. Add `GROQ_API_KEY` in Project Settings -> Environment Variables.
4. Deploy.

## Troubleshooting

- If WASM/module resolution fails, verify `next.config.mjs` has:
  - WASM asset rule (`test: /\.wasm$/`)
  - `asyncWebAssembly` experiment
  - aliases for `sharp` and `onnxruntime-node`
- First desktop run can be slow due to model download and shader compilation.
- Keep the tab active during long transcriptions.
