import { afterEach, describe, expect, it, vi } from "vitest";
import {
  isAllowedBlobPathname,
  isAllowedOrigin,
  isManagedBlobUrl,
  isValidAudioFileName,
  isValidAudioMimeType,
  sanitizeUploadFileName,
} from "./cloud-transcription";

describe("cloud transcription validation", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("accepts same-origin requests and rejects cross-origin requests", () => {
    const sameOrigin = new Request("https://audio-transcription.app/api/transcribe", {
      headers: {
        host: "audio-transcription.app",
        origin: "https://audio-transcription.app",
      },
    });
    const crossOrigin = new Request("https://audio-transcription.app/api/transcribe", {
      headers: {
        host: "audio-transcription.app",
        origin: "https://example.com",
      },
    });
    const spoofedHost = new Request("https://attacker.example/api/transcribe", {
      headers: {
        host: "attacker.example",
        origin: "https://attacker.example",
      },
    });

    expect(isAllowedOrigin(sameOrigin)).toBe(true);
    expect(isAllowedOrigin(crossOrigin)).toBe(false);
    expect(isAllowedOrigin(spoofedHost)).toBe(false);

    vi.stubEnv("VERCEL_URL", "preview-audio.vercel.app");
    const previewOrigin = new Request("https://preview-audio.vercel.app/api/transcribe", {
      headers: { origin: "https://preview-audio.vercel.app" },
    });
    expect(isAllowedOrigin(previewOrigin)).toBe(true);
  });

  it("sanitizes untrusted upload filenames", () => {
    expect(sanitizeUploadFileName("  ../My Meeting (Final).M4A  ")).toBe("my-meeting-final.m4a");
    expect(sanitizeUploadFileName("...")).toBe("audio.wav");
  });

  it("restricts managed paths and Blob URLs", () => {
    expect(isAllowedBlobPathname("transcriptions/2026-07-14/job/chunk-0.ogg")).toBe(true);
    expect(isAllowedBlobPathname("transcriptions/../private.txt")).toBe(false);
    expect(isAllowedBlobPathname("other/chunk-0.ogg")).toBe(false);
    expect(isManagedBlobUrl("https://store.public.blob.vercel-storage.com/transcriptions/a.ogg"))
      .toBe(true);
    expect(isManagedBlobUrl("http://store.public.blob.vercel-storage.com/transcriptions/a.ogg"))
      .toBe(false);
    expect(isManagedBlobUrl("https://blob.vercel-storage.com.example.com/a.ogg")).toBe(false);
  });

  it("accepts supported audio MIME types only", () => {
    expect(isValidAudioMimeType("audio/mpeg")).toBe(true);
    expect(isValidAudioMimeType("video/mp4")).toBe(true);
    expect(isValidAudioMimeType("text/html")).toBe(false);
    expect(isValidAudioFileName("meeting.M4A")).toBe(true);
    expect(isValidAudioFileName("meeting.exe")).toBe(false);
    expect(isValidAudioFileName("meeting")).toBe(false);
  });
});
