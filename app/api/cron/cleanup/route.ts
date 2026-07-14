import { timingSafeEqual } from "node:crypto";
import { del, list } from "@vercel/blob";
import { NextResponse } from "next/server";
import { BLOB_UPLOAD_PATH_PREFIX } from "@/lib/cloud-transcription";
import { TRANSCRIPTION_SESSION_TTL_MS } from "@/lib/transcription-session";

export const runtime = "nodejs";
export const maxDuration = 60;

const CLEANUP_GRACE_PERIOD_MS = 60 * 60 * 1000;
const MAX_LIST_PAGES = 100;

function isAuthorized(request: Request): boolean {
  const cronSecret = process.env.CRON_SECRET;
  const authorization = request.headers.get("authorization");
  if (!cronSecret || !authorization) return false;

  const expected = Buffer.from(`Bearer ${cronSecret}`);
  const provided = Buffer.from(authorization);
  return provided.length === expected.length && timingSafeEqual(provided, expected);
}

export async function GET(request: Request): Promise<NextResponse> {
  if (!isAuthorized(request)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const staleBefore = Date.now() - TRANSCRIPTION_SESSION_TTL_MS - CLEANUP_GRACE_PERIOD_MS;
  let cursor: string | undefined;
  let deleted = 0;
  let scanned = 0;

  for (let page = 0; page < MAX_LIST_PAGES; page += 1) {
    const result = await list({
      cursor,
      limit: 1_000,
      prefix: BLOB_UPLOAD_PATH_PREFIX,
    });
    scanned += result.blobs.length;

    const stalePathnames = result.blobs
      .filter((blob) => blob.uploadedAt.getTime() < staleBefore)
      .map((blob) => blob.pathname);

    if (stalePathnames.length > 0) {
      await del(stalePathnames);
      deleted += stalePathnames.length;
    }

    if (!result.hasMore || !result.cursor) break;
    cursor = result.cursor;
  }

  return NextResponse.json(
    { deleted, scanned },
    { headers: { "Cache-Control": "no-store" } },
  );
}