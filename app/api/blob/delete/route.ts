import { del } from "@vercel/blob";
import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import {
  isAllowedOrigin,
  isManagedBlobUrl,
  TRANSCRIPTION_SESSION_HEADER,
} from "@/lib/cloud-transcription";
import { assertChunkPathname, getTranscriptionSessionToken } from "@/lib/transcription-session";

type DeleteBlobRequest = {
  blobUrl?: string;
};

export async function POST(request: Request): Promise<NextResponse> {
  if (!isAllowedOrigin(request)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const botResponse = await denyBotTraffic();
  if (botResponse) return botResponse;

  let payload: DeleteBlobRequest;
  try {
    payload = (await request.json()) as DeleteBlobRequest;
  } catch {
    return NextResponse.json({ error: "Invalid delete payload." }, { status: 400 });
  }

  const blobUrl = typeof payload.blobUrl === "string" ? payload.blobUrl : "";
  if (!isManagedBlobUrl(blobUrl)) {
    return NextResponse.json({ error: "Invalid blob URL." }, { status: 400 });
  }

  const pathname = new URL(blobUrl).pathname.slice(1);
  const sessionToken = getTranscriptionSessionToken(request);
  if (!assertChunkPathname(sessionToken, pathname)) {
    return NextResponse.json(
      {
        error: `Missing or invalid ${TRANSCRIPTION_SESSION_HEADER} header.`,
      },
      { status: 401 },
    );
  }

  try {
    await del(blobUrl);
  } catch {
    // Cleanup is best-effort so temporary uploads do not block the user flow.
  }

  return NextResponse.json({ ok: true });
}
