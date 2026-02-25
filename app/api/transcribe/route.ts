import { NextResponse } from "next/server";

const LANGUAGE_CODES: Record<string, string> = {
  english: "en",
  turkish: "tr",
  spanish: "es",
  french: "fr",
  german: "de",
  italian: "it",
  portuguese: "pt",
  russian: "ru",
  arabic: "ar",
  hindi: "hi",
  japanese: "ja",
  korean: "ko",
};

const MAX_AUDIO_FILE_BYTES = 4_500_000;
const RATE_LIMIT_WINDOW_MS = 60_000;
// Mobile long-audio transcription is chunked into many API calls.
// Keep abuse protection, but allow sustained chunk uploads.
const MAX_REQUESTS_PER_WINDOW = 90;
const ALLOWED_AUDIO_TYPES = new Set([
  "audio/mpeg",
  "audio/wav",
  "audio/x-wav",
  "audio/mp4",
  "audio/x-m4a",
  "audio/ogg",
  "audio/flac",
  "audio/aac",
  "audio/webm",
  "audio/opus",
  "video/mp4",
  "application/octet-stream",
]);
const RATE_LIMIT_STORE = new Map<string, { windowStart: number; count: number }>();

function jsonError(
  message: string,
  status: number,
  headers: Record<string, string> = {},
): NextResponse {
  return NextResponse.json(
    { error: message },
    {
      status,
      headers: {
        "Cache-Control": "no-store",
        ...headers,
      },
    },
  );
}

function getRequestIp(req: Request): string {
  const forwardedFor = req.headers.get("x-forwarded-for");
  if (forwardedFor) {
    return forwardedFor.split(",")[0]?.trim() || "unknown";
  }
  const realIp = req.headers.get("x-real-ip");
  if (realIp) return realIp;
  return "unknown";
}

function cleanupRateLimitStore(now: number): void {
  for (const [ip, state] of RATE_LIMIT_STORE) {
    if (now - state.windowStart > RATE_LIMIT_WINDOW_MS * 2) {
      RATE_LIMIT_STORE.delete(ip);
    }
  }
}

function isRateLimited(ip: string, now: number): boolean {
  const state = RATE_LIMIT_STORE.get(ip);
  if (!state || now - state.windowStart > RATE_LIMIT_WINDOW_MS) {
    RATE_LIMIT_STORE.set(ip, { windowStart: now, count: 1 });
    return false;
  }

  state.count += 1;
  RATE_LIMIT_STORE.set(ip, state);
  return state.count > MAX_REQUESTS_PER_WINDOW;
}

function getRetryAfterSeconds(ip: string, now: number): number {
  const state = RATE_LIMIT_STORE.get(ip);
  if (!state) {
    return Math.ceil(RATE_LIMIT_WINDOW_MS / 1000);
  }

  const resetAt = state.windowStart + RATE_LIMIT_WINDOW_MS;
  const msUntilReset = Math.max(resetAt - now, 1_000);
  return Math.ceil(msUntilReset / 1_000);
}

function toHostname(urlValue: string | null): string | null {
  if (!urlValue) return null;
  try {
    return new URL(urlValue).hostname;
  } catch {
    return null;
  }
}

function isAllowedOrigin(req: Request): boolean {
  const headerHost = req.headers.get("x-forwarded-host") ?? req.headers.get("host");
  const requestHost = headerHost?.split(":")[0]?.toLowerCase() ?? null;
  const knownHosts = new Set<string>([
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "audio-transcription.app",
    "www.audio-transcription.app",
  ]);

  if (requestHost) knownHosts.add(requestHost);

  const envUrls = [process.env.APP_URL, process.env.NEXT_PUBLIC_SITE_URL];
  for (const envUrl of envUrls) {
    const host = toHostname(envUrl ?? null);
    if (host) knownHosts.add(host.toLowerCase());
  }

  const originHost = toHostname(req.headers.get("origin"));
  const refererHost = toHostname(req.headers.get("referer"));
  const sourceHost = originHost ?? refererHost;

  if (!sourceHost) {
    return process.env.NODE_ENV !== "production";
  }

  return knownHosts.has(sourceHost.toLowerCase());
}

function isValidAudioMimeType(type: string): boolean {
  if (!type) return true;
  return type.startsWith("audio/") || ALLOWED_AUDIO_TYPES.has(type.toLowerCase());
}

type GroqSegment = { text: string; start: number; end: number };
type GroqResponse = { text?: string; segments?: GroqSegment[] };

export async function POST(req: Request) {
  try {
    if (!isAllowedOrigin(req)) {
      return jsonError("Forbidden", 403);
    }

    const now = Date.now();
    if (RATE_LIMIT_STORE.size > 500) cleanupRateLimitStore(now);

    const ip = getRequestIp(req);
    if (isRateLimited(ip, now)) {
      const retryAfterSeconds = getRetryAfterSeconds(ip, now);
      return jsonError(
        "Too many requests. Please try again shortly.",
        429,
        {
          "Retry-After": String(retryAfterSeconds),
          "X-RateLimit-Limit": String(MAX_REQUESTS_PER_WINDOW),
          "X-RateLimit-Window": String(Math.ceil(RATE_LIMIT_WINDOW_MS / 1_000)),
          "X-RateLimit-Remaining": "0",
        },
      );
    }

    const formData = await req.formData();
    const fileEntry = formData.get("file");
    const languageEntry = formData.get("language");

    if (!(fileEntry instanceof File)) {
      return jsonError("No audio file provided.", 400);
    }
    if (fileEntry.size <= 0) {
      return jsonError("Uploaded audio file is empty.", 400);
    }
    if (fileEntry.size > MAX_AUDIO_FILE_BYTES) {
      return jsonError("Audio payload is too large.", 413);
    }
    if (!isValidAudioMimeType(fileEntry.type)) {
      return jsonError("Unsupported audio MIME type.", 415);
    }

    const language = typeof languageEntry === "string" ? languageEntry : "";
    if (language && language !== "auto" && !LANGUAGE_CODES[language]) {
      return jsonError("Unsupported language option.", 400);
    }

    if (!process.env.GROQ_API_KEY) {
      return jsonError("GROQ_API_KEY is not configured on the server.", 500);
    }

    const groqFormData = new FormData();
    groqFormData.append("file", fileEntry);
    groqFormData.append("model", "whisper-large-v3");
    groqFormData.append("response_format", "verbose_json");

    if (language && language !== "auto") {
      groqFormData.append("language", LANGUAGE_CODES[language]);
    }

    const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: groqFormData,
      signal: AbortSignal.timeout(45_000),
      cache: "no-store",
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage =
        errorData?.error?.message || "Failed to transcribe audio on Groq.";
      return jsonError(errorMessage, response.status);
    }

    const data = (await response.json()) as GroqResponse;
    return NextResponse.json(
      {
        text: data.text ?? "",
        segments:
          data.segments?.map((segment) => ({
            text: segment.text,
            start: segment.start,
            end: segment.end,
          })) ?? [],
      },
      {
        headers: { "Cache-Control": "no-store" },
      },
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      return jsonError("Cloud transcription timed out. Please try again.", 504);
    }

    console.error("Groq transcription error:", error);
    return jsonError("Internal Server Error", 500);
  }
}
