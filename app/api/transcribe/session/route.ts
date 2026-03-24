import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import { isAllowedOrigin } from "@/lib/cloud-transcription";
import {
  MAX_TRANSCRIPTION_SESSION_CHUNKS,
  createTranscriptionSession,
} from "@/lib/transcription-session";

const SESSION_WINDOW_MS = 30 * 60 * 1000;
const MAX_SESSION_STARTS_PER_WINDOW = 10;
const SESSION_START_STORE = new Map<string, { count: number; windowStart: number }>();

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

function cleanupSessionStartStore(now: number): void {
  for (const [ip, state] of SESSION_START_STORE) {
    if (now - state.windowStart > SESSION_WINDOW_MS * 2) {
      SESSION_START_STORE.delete(ip);
    }
  }
}

function isSessionStartRateLimited(ip: string, now: number): boolean {
  const state = SESSION_START_STORE.get(ip);
  if (!state || now - state.windowStart > SESSION_WINDOW_MS) {
    SESSION_START_STORE.set(ip, { count: 1, windowStart: now });
    return false;
  }

  state.count += 1;
  SESSION_START_STORE.set(ip, state);
  return state.count > MAX_SESSION_STARTS_PER_WINDOW;
}

export async function POST(request: Request): Promise<NextResponse> {
  if (!isAllowedOrigin(request)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const botResponse = await denyBotTraffic();
  if (botResponse) return botResponse;

  const now = Date.now();
  if (SESSION_START_STORE.size > 500) cleanupSessionStartStore(now);

  const ip = getRequestIp(request);
  if (isSessionStartRateLimited(ip, now)) {
    return NextResponse.json(
      { error: "Too many transcription sessions started. Please wait a bit and try again." },
      { status: 429, headers: { "Retry-After": String(Math.ceil(SESSION_WINDOW_MS / 1000)) } },
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

  return NextResponse.json(createTranscriptionSession(totalChunks, fileName), {
    headers: { "Cache-Control": "no-store" },
  });
}
