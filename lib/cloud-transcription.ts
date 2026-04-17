export const BLOB_UPLOAD_ACCESS =
  process.env.NEXT_PUBLIC_BLOB_UPLOAD_ACCESS === "public" ? "public" : "private";
export const USE_DIRECT_BLOB_URL_HANDOFF = BLOB_UPLOAD_ACCESS === "public";
// Public mode: blob is uploaded directly to Blob storage (no 4.5 MB function
// body limit) and Groq fetches the URL directly. Larger chunks reduce the
// number of Groq API calls, which makes long recordings far less likely to
// stall on per-minute rate limits.
// Private mode: the audio has to be proxied through the function body → keep
// the conservative 4.5 MB cap that fits under Vercel's serverless body limit.
export const MAX_AUDIO_FILE_BYTES =
  BLOB_UPLOAD_ACCESS === "public" ? 24 * 1024 * 1024 : 4_500_000;
export const BLOB_UPLOAD_PATH_PREFIX = "transcriptions/";
export const TRANSCRIPTION_SESSION_HEADER = "x-transcription-session";
export const ALLOWED_AUDIO_CONTENT_TYPES = [
  "audio/mpeg",
  "audio/wav",
  "audio/x-wav",
  "audio/mp4",
  "audio/x-m4a",
  "audio/ogg",
  "audio/flac",
  "audio/aac",
  "audio/webm",
  "audio/opus",
  "video/mp4",
  "application/octet-stream",
] as const;

const ALLOWED_AUDIO_TYPE_SET = new Set<string>(ALLOWED_AUDIO_CONTENT_TYPES);
const BLOB_HOST_SUFFIX = ".blob.vercel-storage.com";

export function toHostname(urlValue: string | null): string | null {
  if (!urlValue) return null;

  try {
    return new URL(urlValue).hostname;
  } catch {
    return null;
  }
}

export function isAllowedOrigin(req: Request): boolean {
  const headerHost = req.headers.get("x-forwarded-host") ?? req.headers.get("host");
  const requestHost = headerHost?.split(":")[0]?.toLowerCase() ?? null;
  const knownHosts = new Set<string>([
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "audio-transcription.app",
    "www.audio-transcription.app",
  ]);

  if (requestHost) knownHosts.add(requestHost);

  const envUrls = [process.env.APP_URL, process.env.NEXT_PUBLIC_SITE_URL];
  for (const envUrl of envUrls) {
    const host = toHostname(envUrl ?? null);
    if (host) knownHosts.add(host.toLowerCase());
  }

  const originHost = toHostname(req.headers.get("origin"));
  const refererHost = toHostname(req.headers.get("referer"));
  const sourceHost = originHost ?? refererHost;

  if (!sourceHost) {
    return process.env.NODE_ENV !== "production";
  }

  return knownHosts.has(sourceHost.toLowerCase());
}

export function isValidAudioMimeType(type: string): boolean {
  if (!type) return true;
  return type.startsWith("audio/") || ALLOWED_AUDIO_TYPE_SET.has(type.toLowerCase());
}

export function sanitizeUploadFileName(fileName: string): string {
  const normalized = fileName.trim().toLowerCase();
  const sanitized = normalized
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^\.+/, "")
    .slice(0, 96);

  return sanitized || "audio.wav";
}

export function isAllowedBlobPathname(pathname: string): boolean {
  return pathname.startsWith(BLOB_UPLOAD_PATH_PREFIX) && !pathname.includes("..");
}

export function isManagedBlobUrl(blobUrl: string): boolean {
  try {
    const parsedUrl = new URL(blobUrl);
    return parsedUrl.protocol === "https:" && parsedUrl.hostname.endsWith(BLOB_HOST_SUFFIX);
  } catch {
    return false;
  }
}
