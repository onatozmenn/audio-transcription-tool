import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import {
  BLOB_UPLOAD_PATH_PREFIX,
  TRANSCRIPTION_SESSION_HEADER,
} from "@/lib/cloud-transcription";

export const MAX_TRANSCRIPTION_SESSION_CHUNKS = 96;
const SESSION_TTL_MS = 4 * 60 * 60 * 1000;

type TranscriptionChunkClaims = {
  exp: number;
  jobId: string;
  chunkIndex: number;
  totalChunks: number;
  pathname: string;
};

export type TranscriptionChunkGrant = {
  chunkIndex: number;
  pathname: string;
  token: string;
};

export type TranscriptionSessionPayload = {
  expiresAt: string;
  grants: TranscriptionChunkGrant[];
  jobId: string;
};

function getSessionSecret(): string {
  const secret = process.env.TRANSCRIPTION_SESSION_SECRET ?? process.env.GROQ_API_KEY;
  if (!secret) {
    throw new Error("TRANSCRIPTION_SESSION_SECRET or GROQ_API_KEY must be configured.");
  }

  return secret;
}

function signPayload(encodedPayload: string): string {
  return createHmac("sha256", getSessionSecret())
    .update(encodedPayload)
    .digest("base64url");
}

function encodeClaims(claims: TranscriptionChunkClaims): string {
  const encodedPayload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const signature = signPayload(encodedPayload);
  return `${encodedPayload}.${signature}`;
}

function decodeClaims(token: string): TranscriptionChunkClaims | null {
  const [encodedPayload, providedSignature] = token.split(".");
  if (!encodedPayload || !providedSignature) return null;

  const expectedSignature = signPayload(encodedPayload);
  const providedBuffer = Buffer.from(providedSignature);
  const expectedBuffer = Buffer.from(expectedSignature);
  if (providedBuffer.length !== expectedBuffer.length) return null;
  if (!timingSafeEqual(providedBuffer, expectedBuffer)) return null;

  try {
    return JSON.parse(
      Buffer.from(encodedPayload, "base64url").toString("utf8"),
    ) as TranscriptionChunkClaims;
  } catch {
    return null;
  }
}

function extractSafeExtension(fileName: string): string {
  const rawExtension = fileName.split(".").pop()?.trim().toLowerCase() ?? "";
  if (!rawExtension || !/^[a-z0-9]{1,10}$/.test(rawExtension)) {
    return ".bin";
  }

  return `.${rawExtension}`;
}

function buildChunkPathname(jobId: string, chunkIndex: number, fileName: string): string {
  const dateKey = new Date().toISOString().slice(0, 10);
  const fileExtension = extractSafeExtension(fileName);
  return `${BLOB_UPLOAD_PATH_PREFIX}${dateKey}/${jobId}/chunk-${chunkIndex}${fileExtension}`;
}

function normalizePathname(pathname: string): string {
  return pathname.replace(/^\/+/, "");
}

export function createTranscriptionSession(totalChunks: number, fileName: string): TranscriptionSessionPayload {
  const expiresAt = Date.now() + SESSION_TTL_MS;
  const jobId = randomUUID();
  const grants: TranscriptionChunkGrant[] = [];

  for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
    const pathname = buildChunkPathname(jobId, chunkIndex, fileName);
    const token = encodeClaims({
      exp: expiresAt,
      jobId,
      chunkIndex,
      totalChunks,
      pathname,
    });

    grants.push({ chunkIndex, pathname, token });
  }

  return {
    expiresAt: new Date(expiresAt).toISOString(),
    grants,
    jobId,
  };
}

export function getTranscriptionSessionToken(req: Request): string | null {
  const token = req.headers.get(TRANSCRIPTION_SESSION_HEADER);
  return token?.trim() || null;
}

export function verifyTranscriptionSessionToken(
  token: string | null,
): TranscriptionChunkClaims | null {
  if (!token) return null;

  const claims = decodeClaims(token);
  if (!claims) return null;
  if (!Number.isInteger(claims.chunkIndex) || claims.chunkIndex < 0) return null;
  if (!Number.isInteger(claims.totalChunks) || claims.totalChunks < 1) return null;
  if (typeof claims.pathname !== "string" || !claims.pathname.startsWith(BLOB_UPLOAD_PATH_PREFIX)) {
    return null;
  }
  if (typeof claims.jobId !== "string" || claims.jobId.length < 8) return null;
  if (Date.now() > claims.exp) return null;

  return claims;
}

export function assertChunkPathname(
  token: string | null,
  pathname: string,
): TranscriptionChunkClaims | null {
  const claims = verifyTranscriptionSessionToken(token);
  if (!claims || claims.pathname !== normalizePathname(pathname)) return null;
  return claims;
}
