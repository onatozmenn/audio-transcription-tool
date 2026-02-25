<p align="center">
  <img src="public/logo.svg" alt="Audio Transcription Tool logo" width="96" />
</p>

<h1 align="center">Audio Transcription Tool</h1>

<p align="center">
  Privacy-first speech-to-text for real users: local on desktop, secure cloud fallback on mobile.
</p>

<p align="center">
  <a href="https://audio-transcription.app"><strong>Live Demo</strong></a>
  <br />
  <sub>Free, no sign-up, multilingual, and optimized for long recordings.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/onatozmenn/audio-transcription-tool?style=flat-square" alt="GitHub stars" />
  <img src="https://img.shields.io/github/forks/onatozmenn/audio-transcription-tool?style=flat-square" alt="GitHub forks" />
  <img src="https://img.shields.io/github/license/onatozmenn/audio-transcription-tool?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/TypeScript-Strict-3178c6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" />
</p>

## Screenshots

| Desktop | Mobile |
| --- | --- |
| ![Desktop screenshot](docs/screenshots/home-desktop.png) | ![Mobile screenshot](docs/screenshots/home-mobile.png) |

## Why This Project

Most transcription tools force users into uploads, accounts, and unclear privacy tradeoffs. This app keeps the default path private and fast:

- Desktop: transcription runs in-browser with Whisper Small (WebGPU/WASM).
- Mobile: secure server route to Groq Whisper Large V3 for reliability.
- Long recordings: chunked processing, retry/backoff, and progress-aware UX.

If this project is useful, please star the repo. It helps a lot.

## Core Features

- Free audio-to-text workflow (no sign-up wall).
- Smart mobile fallback when local inference is not practical.
- Chunk-based processing for long recordings.
- Export options: plain text, timestamped text, JSON.
- English + Turkish route/SEO support.
- Privacy page and clear data-handling model.

## How It Works

### 1. Desktop path (local inference)

- Audio is decoded in browser.
- A Web Worker loads `Xenova/whisper-small` via `@huggingface/transformers`.
- Worker reports download/transcription progress and partial text.
- No server call is required for desktop transcription.

### 2. Mobile path (cloud fallback)

- Audio is chunked into upload-safe WAV parts.
- Chunks are sent to `/api/transcribe`.
- Server forwards requests to Groq `whisper-large-v3`.
- Responses are merged into a full transcript with segment timestamps.

### 3. Long-audio reliability protections

- Larger chunk duration to reduce request count.
- API rate limit tuned for chunk workflows.
- `429`/`5xx` retry with backoff and `Retry-After` handling.
- Abort-safe wait/retry logic for cancellation.

## Architecture

```mermaid
flowchart LR
    A[User Upload] --> B{Device Type}
    B -->|Desktop| C[Decode Audio in Browser]
    C --> D[Web Worker + Whisper Small]
    D --> E[Progress + Partial Transcript]
    E --> F[Final Transcript + Export]

    B -->|Mobile| G[Chunk + WAV Encode]
    G --> H[/api/transcribe]
    H --> I[Groq whisper-large-v3]
    I --> J[Chunk Transcript + Segments]
    J --> K[Merge + Offset Timestamps]
    K --> F
```

## Tech Stack

- Next.js App Router
- TypeScript (strict)
- Tailwind CSS
- Web Workers
- `@huggingface/transformers`
- Groq Audio Transcription API

## Quick Start

### Prerequisites

- Node.js 20+
- npm or pnpm

### Install

```bash
npm install
# or
pnpm install
```

### Run

```bash
npm run dev
# or
pnpm dev
```

Open `http://localhost:3000`.

### Production check

```bash
npm run build && npm run start
# or
pnpm build && pnpm start
```

## Environment Variables

Create `.env.local`:

```bash
GROQ_API_KEY=your_groq_api_key
```

Notes:

- `GROQ_API_KEY` is required for mobile cloud fallback.
- Desktop local transcription works without cloud calls.

## Deployment (Vercel)

1. Push this repo to GitHub.
2. Import project in Vercel.
3. Add `GROQ_API_KEY` in Project Settings -> Environment Variables.
4. Deploy.

## Privacy Model

- Desktop: local-only processing in browser runtime.
- Mobile: audio is processed via secure cloud fallback endpoint.
- No account or user identity flow is required by the app.

See full policy: [Privacy Policy](https://audio-transcription.app/privacy)

## Troubleshooting

- WASM issues: verify `next.config.mjs` has async WASM + required aliases.
- First desktop run can be slow (model download + shader compile).
- For long recordings, keep the tab open and active during processing.

## Contributing

PRs and issues are welcome. If you propose a change, include:

- Problem statement
- Before/after behavior
- Test or reproduction notes

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

Screenshots and logo in this repository are distributed under the same license unless stated otherwise.
