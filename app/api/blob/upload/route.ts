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
  let body: HandleUploadBody;
  try {
    body = (await request.json()) as HandleUploadBody;
  } catch {
    return NextResponse.json({ error: "Invalid upload payload." }, { status: 400 });
  }

  let sessionToken: string | null = null;
  if (body.type === "blob.generate-client-token") {
    if (!isAllowedOrigin(request)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const botResponse = await denyBotTraffic();
    if (botResponse) return botResponse;

    sessionToken = getTranscriptionSessionToken(request);
    if (!assertChunkPathname(sessionToken, body.payload.pathname)) {
      return NextResponse.json(
        {
          error: `Missing or invalid ${TRANSCRIPTION_SESSION_HEADER} header.`,
        },
        { status: 401 },
      );
    }
  } else {
    sessionToken = body.payload.tokenPayload?.trim() || null;
    if (!assertChunkPathname(sessionToken, body.payload.blob.pathname)) {
      return NextResponse.json({ error: "Invalid upload completion payload." }, { status: 401 });
    }
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
          // Session grants already reserve a unique job/chunk pathname, so
          // Blob must keep that exact pathname for later session validation.
          addRandomSuffix: false,
          allowOverwrite: true,
          tokenPayload: sessionToken,
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
