import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";
import { denyBotTraffic } from "@/lib/bot-protection";
import {
  ALLOWED_AUDIO_CONTENT_TYPES,
  MAX_AUDIO_FILE_BYTES,
  isAllowedBlobPathname,
  isAllowedOrigin,
  TRANSCRIPTION_SESSION_HEADER,
} from "@/lib/cloud-transcription";
import { assertChunkPathname, getTranscriptionSessionToken } from "@/lib/transcription-session";

export async function POST(request: Request): Promise<NextResponse> {
  if (!isAllowedOrigin(request)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const botResponse = await denyBotTraffic();
  if (botResponse) return botResponse;

  let body: HandleUploadBody;
  try {
    body = (await request.json()) as HandleUploadBody;
  } catch {
    return NextResponse.json({ error: "Invalid upload payload." }, { status: 400 });
  }

  const pathname =
    body.type === "blob.generate-client-token" ? body.payload.pathname : "";
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
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        if (!isAllowedBlobPathname(pathname)) {
          throw new Error("Unsupported upload path.");
        }

        return {
          allowedContentTypes: [...ALLOWED_AUDIO_CONTENT_TYPES],
          maximumSizeInBytes: MAX_AUDIO_FILE_BYTES,
          addRandomSuffix: true,
        };
      },
    });

    return NextResponse.json(jsonResponse);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to authorize upload.",
      },
      { status: 400 },
    );
  }
}
