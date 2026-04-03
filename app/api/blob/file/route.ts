import { get } from "@vercel/blob";
import { NextResponse } from "next/server";
import { isAllowedBlobPathname } from "@/lib/cloud-transcription";
import { assertChunkPathname } from "@/lib/transcription-session";

export const runtime = "nodejs";

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const pathname = searchParams.get("pathname")?.trim() ?? "";
  const token = searchParams.get("token")?.trim() ?? "";

  if (!pathname || !isAllowedBlobPathname(pathname)) {
    return NextResponse.json({ error: "Invalid blob pathname." }, { status: 400 });
  }

  if (!assertChunkPathname(token, pathname)) {
    return NextResponse.json({ error: "Invalid blob access token." }, { status: 401 });
  }

  try {
    const result = await get(pathname, { access: "private" });

    if (!result || result.statusCode !== 200 || !result.stream) {
      return NextResponse.json({ error: "Blob not found." }, { status: 404 });
    }

    return new NextResponse(result.stream, {
      headers: {
        "Cache-Control": "private, no-store",
        "Content-Length": String(result.blob.size),
        "Content-Type": result.blob.contentType || "application/octet-stream",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    return NextResponse.json({ error: "Blob could not be streamed." }, { status: 404 });
  }
}
