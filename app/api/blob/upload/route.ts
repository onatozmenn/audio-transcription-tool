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
    const parsed = (await request.json()) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return NextResponse.json({ error: "Invalid upload payload." }, { status: 400 });
    }
    body = parsed as HandleUploadBody;
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

    const pathname = body.payload?.pathname;
    if (typeof pathname !== "string") {
      return NextResponse.json({ error: "Invalid upload pathname." }, { status: 400 });
    }

    sessionToken = getTranscriptionSessionToken(request);
    if (!assertChunkPathname(sessionToken, pathname)) {
      return NextResponse.json(
        {
          error: `Missing or invalid ${TRANSCRIPTION_SESSION_HEADER} header.`,
        },
        { status: 401 },
      );
    }
  } else {
    const tokenPayload = body.payload?.tokenPayload;
    const pathname = body.payload?.blob?.pathname;
    sessionToken = typeof tokenPayload === "string" ? tokenPayload.trim() || null : null;
    if (typeof pathname !== "string" || !assertChunkPathname(sessionToken, pathname)) {
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
    console.error("Blob upload authorization failed:", error);
    return NextResponse.json(
      { error: "Failed to authorize upload." },
      { status: 400 },
    );
  }
}
