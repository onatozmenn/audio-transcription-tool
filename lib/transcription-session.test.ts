import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  MAX_TRANSCRIPTION_SESSION_CHUNKS,
  TRANSCRIPTION_SESSION_TTL_MS,
  assertChunkPathname,
  createTranscriptionSession,
  verifyTranscriptionSessionToken,
} from "./transcription-session";

describe("transcription sessions", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-14T12:00:00Z"));
    vi.stubEnv("TRANSCRIPTION_SESSION_SECRET", "test-session-secret-with-enough-entropy");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
  });

  it("creates a distinct grant bound to every reserved pathname", () => {
    const session = createTranscriptionSession(2, "Meeting Audio.M4A");

    expect(session.grants).toHaveLength(2);
    expect(session.grants[0].pathname).toMatch(
      /^transcriptions\/2026-07-14\/[0-9a-f-]+\/chunk-0\.m4a$/,
    );
    expect(session.grants[1].pathname).toContain("/chunk-1.m4a");
    expect(session.grants[0].token).not.toBe(session.grants[1].token);

    const claims = verifyTranscriptionSessionToken(session.grants[0].token);
    expect(claims).toMatchObject({
      chunkIndex: 0,
      jobId: session.jobId,
      pathname: session.grants[0].pathname,
      totalChunks: 2,
    });
  });

  it("rejects token tampering and pathname substitution", () => {
    const session = createTranscriptionSession(1, "audio.wav");
    const grant = session.grants[0];
    const tamperedToken = `${grant.token.slice(0, -1)}x`;

    expect(verifyTranscriptionSessionToken(tamperedToken)).toBeNull();
    expect(assertChunkPathname(grant.token, `${grant.pathname}.other`)).toBeNull();
    expect(assertChunkPathname(grant.token, `/${grant.pathname}`)?.jobId).toBe(session.jobId);
  });

  it("rejects expired grants", () => {
    const session = createTranscriptionSession(1, "audio.ogg");
    vi.advanceTimersByTime(TRANSCRIPTION_SESSION_TTL_MS + 1);

    expect(verifyTranscriptionSessionToken(session.grants[0].token)).toBeNull();
  });

  it("enforces chunk and filename invariants at the session boundary", () => {
    expect(() => createTranscriptionSession(0, "audio.wav")).toThrow(RangeError);
    expect(() => createTranscriptionSession(MAX_TRANSCRIPTION_SESSION_CHUNKS + 1, "audio.wav"))
      .toThrow(RangeError);
    expect(() => createTranscriptionSession(1, "   ")).toThrow(/file name/i);
    expect(() => createTranscriptionSession(1, "payload.exe")).toThrow(/extension/i);
  });
});
