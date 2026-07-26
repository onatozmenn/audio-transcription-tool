import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import { isAllowedOrigin, isValidAudioFileName } from "@/lib/cloud-transcription";
import {
  MAX_TRANSCRIPTION_SESSION_CHUNKS,
  createTranscriptionSession,
} from "@/lib/transcription-session";
import { checkRateLimit } from "@/lib/rate-limit";

const SESSION_WINDOW_MS = 30 * 60 * 1000;
// Mobile carriers put many subscribers behind a single CGNAT address, so a
// per-IP budget that suits one desktop user locks out an entire cell.
const MAX_SESSION_STARTS_PER_WINDOW = 30;

type CreateSessionRequest = {
  fileName?: string;
  totalChunks?: number;
};

function getRequestIp(req: Request): string {
  const forwardedFor = req.headers.get("x-forwarded-for");
  if (forwardedFor) {
    return forwardedFor.split(",")[0]?.trim() || "unknown";
  }

  return req.headers.get("x-real-ip") ?? "unknown";
}

export async function POST(request: Request): Promise<NextResponse> {
  if (!isAllowedOrigin(request)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const botResponse = await denyBotTraffic();
  if (botResponse) return botResponse;

  const ip = getRequestIp(request);
  const rateLimit = await checkRateLimit({
    identifier: ip,
    limit: MAX_SESSION_STARTS_PER_WINDOW,
    namespace: "session",
    windowMs: SESSION_WINDOW_MS,
  });
  if (!rateLimit.available) {
    return NextResponse.json(
      { error: "Rate limit service is temporarily unavailable." },
      { status: 503, headers: { "Retry-After": String(rateLimit.retryAfterSeconds) } },
    );
  }
  if (!rateLimit.allowed) {
    return NextResponse.json(
      { error: "Too many transcription sessions started. Please wait a bit and try again." },
      { status: 429, headers: { "Retry-After": String(rateLimit.retryAfterSeconds) } },
    );
  }

  let body: CreateSessionRequest;
  try {
    body = (await request.json()) as CreateSessionRequest;
  } catch {
    return NextResponse.json({ error: "Invalid session payload." }, { status: 400 });
  }

  const totalChunks = body.totalChunks;
  const fileName = typeof body.fileName === "string" ? body.fileName : "";

  if (typeof totalChunks !== "number" || !Number.isInteger(totalChunks) || totalChunks < 1) {
    return NextResponse.json({ error: "Invalid chunk count." }, { status: 400 });
  }
  if (totalChunks > MAX_TRANSCRIPTION_SESSION_CHUNKS) {
    return NextResponse.json({ error: "Chunk count exceeds the supported session limit." }, { status: 400 });
  }
  if (!fileName.trim()) {
    return NextResponse.json({ error: "Invalid file name." }, { status: 400 });
  }
  if (!isValidAudioFileName(fileName)) {
    return NextResponse.json({ error: "Unsupported audio file extension." }, { status: 415 });
  }

  return NextResponse.json(createTranscriptionSession(totalChunks, fileName), {
    headers: { "Cache-Control": "no-store" },
  });
}
