import { head } from "@vercel/blob";
import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import {
  MAX_AUDIO_FILE_BYTES,
  isAllowedBlobPathname,
  isAllowedOrigin,
  isManagedBlobUrl,
  isValidAudioMimeType,
  TRANSCRIPTION_SESSION_HEADER,
  USE_DIRECT_BLOB_URL_HANDOFF,
} from "@/lib/cloud-transcription";
import { assertChunkPathname, getTranscriptionSessionToken } from "@/lib/transcription-session";

// Long-audio chunks can take ~20-40s on Groq (fetch blob → transcribe → return).
// Default Vercel function timeout (10s hobby / 15s pro) is not enough — bump
// to 60s so individual chunk requests don't get cut off mid-flight.
export const maxDuration = 60;

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

const RATE_LIMIT_WINDOW_MS = 60_000;
// Mobile long-audio transcription is chunked into many API calls.
// Keep abuse protection, but allow sustained chunk uploads.
const MAX_REQUESTS_PER_WINDOW = 90;
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

type GroqSegment = { text: string; start: number; end: number };
type GroqResponse = { text?: string; segments?: GroqSegment[] };
type BlobTranscriptionRequest = { blobUrl?: string; language?: string };

function buildInternalBlobStreamUrl(req: Request, pathname: string, sessionToken: string): string {
  const baseUrl = new URL(req.url);
  const streamUrl = new URL("/api/blob/file", baseUrl);
  streamUrl.searchParams.set("pathname", pathname);
  streamUrl.searchParams.set("token", sessionToken);
  return streamUrl.toString();
}

async function buildGroqFormData(req: Request): Promise<FormData | NextResponse> {
  const contentType = req.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    let body: BlobTranscriptionRequest;
    try {
      body = (await req.json()) as BlobTranscriptionRequest;
    } catch {
      return jsonError("Invalid JSON payload.", 400);
    }

    const blobUrl = typeof body.blobUrl === "string" ? body.blobUrl : "";
    const language = typeof body.language === "string" ? body.language : "";

    if (!isManagedBlobUrl(blobUrl)) {
      return jsonError("Invalid audio upload URL.", 400);
    }
    if (language && language !== "auto" && !LANGUAGE_CODES[language]) {
      return jsonError("Unsupported language option.", 400);
    }

    let blobMeta;
    try {
      blobMeta = await head(blobUrl);
    } catch {
      return jsonError("Uploaded audio could not be found.", 404);
    }

    if (!isAllowedBlobPathname(blobMeta.pathname)) {
      return jsonError("Unsupported audio upload path.", 400);
    }
    const sessionToken = getTranscriptionSessionToken(req);
    const sessionClaims = assertChunkPathname(sessionToken, blobMeta.pathname);
    if (!sessionClaims || !sessionToken) {
      return jsonError(`Missing or invalid ${TRANSCRIPTION_SESSION_HEADER} header.`, 401);
    }
    if (blobMeta.size <= 0) {
      return jsonError("Uploaded audio file is empty.", 400);
    }
    if (blobMeta.size > MAX_AUDIO_FILE_BYTES) {
      return jsonError("Audio payload is too large.", 413);
    }
    if (!isValidAudioMimeType(blobMeta.contentType ?? "")) {
      return jsonError("Unsupported audio MIME type.", 415);
    }

    const groqFormData = new FormData();
    groqFormData.append(
      "url",
      USE_DIRECT_BLOB_URL_HANDOFF
        ? blobMeta.url
        : buildInternalBlobStreamUrl(req, blobMeta.pathname, sessionToken),
    );
    groqFormData.append("model", "whisper-large-v3");
    groqFormData.append("response_format", "verbose_json");

    if (language && language !== "auto") {
      groqFormData.append("language", LANGUAGE_CODES[language]);
    }

    return groqFormData;
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

  const groqFormData = new FormData();
  groqFormData.append("file", fileEntry);
  groqFormData.append("model", "whisper-large-v3");
  groqFormData.append("response_format", "verbose_json");

  if (language && language !== "auto") {
    groqFormData.append("language", LANGUAGE_CODES[language]);
  }

  return groqFormData;
}

export async function POST(req: Request) {
  try {
    if (!isAllowedOrigin(req)) {
      return jsonError("Forbidden", 403);
    }

    const botResponse = await denyBotTraffic();
    if (botResponse) return botResponse;

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

    if (!process.env.GROQ_API_KEY) {
      return jsonError("GROQ_API_KEY is not configured on the server.", 500);
    }

    const groqFormData = await buildGroqFormData(req);
    if (groqFormData instanceof NextResponse) {
      return groqFormData;
    }

    const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: groqFormData,
      signal: AbortSignal.timeout(55_000),
      cache: "no-store",
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage =
        errorData?.error?.message || "Failed to transcribe audio on Groq.";
      const retryAfter = response.headers.get("retry-after");
      return jsonError(
        errorMessage,
        response.status,
        retryAfter ? { "Retry-After": retryAfter } : {},
      );
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
