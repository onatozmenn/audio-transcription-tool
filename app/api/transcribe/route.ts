import { del, head } from "@vercel/blob";
import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import {
  MAX_AUDIO_FILE_BYTES,
  isAllowedBlobPathname,
  isAllowedOrigin,
  isManagedBlobUrl,
  isValidAudioFileName,
  isValidAudioMimeType,
  TRANSCRIPTION_SESSION_HEADER,
  USE_DIRECT_BLOB_URL_HANDOFF,
} from "@/lib/cloud-transcription";
import {
  assertChunkPathname,
  getTranscriptionSessionToken,
  verifyTranscriptionSessionToken,
} from "@/lib/transcription-session";
import { checkRateLimit } from "@/lib/rate-limit";

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

type GroqSegment = { text: string; start: number; end: number };
type GroqResponse = { text?: string; segments?: GroqSegment[] };
type BlobTranscriptionRequest = { blobUrl?: string; language?: string };
type GroqRequest = { cleanupBlobUrl: string | null; formData: FormData };

function buildInternalBlobStreamUrl(req: Request, pathname: string, sessionToken: string): string {
  const baseUrl = new URL(req.url);
  const streamUrl = new URL("/api/blob/file", baseUrl);
  streamUrl.searchParams.set("pathname", pathname);
  streamUrl.searchParams.set("token", sessionToken);
  return streamUrl.toString();
}

async function buildGroqFormData(req: Request): Promise<GroqRequest | NextResponse> {
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

    return { cleanupBlobUrl: blobMeta.url, formData: groqFormData };
  }

  const formData = await req.formData();
  const fileEntry = formData.get("file");
  const languageEntry = formData.get("language");

  const sessionToken = getTranscriptionSessionToken(req);
  if (!verifyTranscriptionSessionToken(sessionToken)) {
    return jsonError(`Missing or invalid ${TRANSCRIPTION_SESSION_HEADER} header.`, 401);
  }

  if (!(fileEntry instanceof File)) {
    return jsonError("No audio file provided.", 400);
  }
  if (!isValidAudioFileName(fileEntry.name)) {
    return jsonError("Unsupported audio file extension.", 415);
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

  return { cleanupBlobUrl: null, formData: groqFormData };
}

export async function POST(req: Request) {
  try {
    if (!isAllowedOrigin(req)) {
      return jsonError("Forbidden", 403);
    }

    const botResponse = await denyBotTraffic();
    if (botResponse) return botResponse;

    const ip = getRequestIp(req);
    const rateLimit = await checkRateLimit({
      identifier: ip,
      limit: MAX_REQUESTS_PER_WINDOW,
      namespace: "transcribe",
      windowMs: RATE_LIMIT_WINDOW_MS,
    });
    if (!rateLimit.available) {
      return jsonError("Rate limit service is temporarily unavailable.", 503, {
        "Retry-After": String(rateLimit.retryAfterSeconds),
      });
    }
    if (!rateLimit.allowed) {
      return jsonError(
        "Too many requests. Please try again shortly.",
        429,
        {
          "Retry-After": String(rateLimit.retryAfterSeconds),
          "X-RateLimit-Limit": String(MAX_REQUESTS_PER_WINDOW),
          "X-RateLimit-Window": String(Math.ceil(RATE_LIMIT_WINDOW_MS / 1_000)),
          "X-RateLimit-Remaining": "0",
        },
      );
    }

    if (!process.env.GROQ_API_KEY) {
      return jsonError("GROQ_API_KEY is not configured on the server.", 500);
    }

    const groqRequest = await buildGroqFormData(req);
    if (groqRequest instanceof NextResponse) {
      return groqRequest;
    }

    const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: groqRequest.formData,
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

    let data: GroqResponse;
    try {
      data = (await response.json()) as GroqResponse;
    } finally {
      if (groqRequest.cleanupBlobUrl) {
        try {
          await del(groqRequest.cleanupBlobUrl);
        } catch (cleanupError) {
          console.error("Server-side Blob cleanup failed:", cleanupError);
        }
      }
    }

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
