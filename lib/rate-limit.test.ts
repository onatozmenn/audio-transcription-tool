import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { checkRateLimit } from "./rate-limit";

describe("rate limit", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("enforces a local fixed window outside production", async () => {
    vi.stubEnv("NODE_ENV", "test");
    vi.stubEnv("UPSTASH_REDIS_REST_URL", "");
    vi.stubEnv("UPSTASH_REDIS_REST_TOKEN", "");
    const namespace = `test-${randomUUID()}`;
    const options = {
      identifier: "203.0.113.10",
      limit: 2,
      namespace,
      windowMs: 60_000,
    };

    expect(await checkRateLimit(options)).toMatchObject({ allowed: true, remaining: 1 });
    expect(await checkRateLimit(options)).toMatchObject({ allowed: true, remaining: 0 });
    expect(await checkRateLimit(options)).toMatchObject({ allowed: false, remaining: 0 });
  });

  it("isolates counters by namespace", async () => {
    vi.stubEnv("NODE_ENV", "test");
    vi.stubEnv("UPSTASH_REDIS_REST_URL", "");
    vi.stubEnv("UPSTASH_REDIS_REST_TOKEN", "");
    const identifier = randomUUID();

    const first = await checkRateLimit({ identifier, limit: 1, namespace: "first", windowMs: 1_000 });
    const second = await checkRateLimit({ identifier, limit: 1, namespace: "second", windowMs: 1_000 });

    expect(first.allowed).toBe(true);
    expect(second.allowed).toBe(true);
  });

  it("degrades to the in-process limiter in production when no shared store is configured", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("RATE_LIMIT_HASH_SECRET", "independent-rate-limit-secret");
    vi.stubEnv("UPSTASH_REDIS_REST_URL", "");
    vi.stubEnv("UPSTASH_REDIS_REST_TOKEN", "");
    const options = {
      identifier: "203.0.113.11",
      limit: 1,
      namespace: `production-${randomUUID()}`,
      windowMs: 1_000,
    };

    await expect(checkRateLimit(options)).resolves.toMatchObject({ allowed: true, available: true });
    await expect(checkRateLimit(options)).resolves.toMatchObject({ allowed: false, available: true });
  });

  it("fails closed in production when the configured store is unreachable", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("RATE_LIMIT_HASH_SECRET", "independent-rate-limit-secret");
    vi.stubEnv("UPSTASH_REDIS_REST_URL", "https://redis.example.com/");
    vi.stubEnv("UPSTASH_REDIS_REST_TOKEN", "test-token");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("connection refused")));

    await expect(checkRateLimit({
      identifier: "203.0.113.13",
      limit: 1,
      namespace: `unreachable-${randomUUID()}`,
      windowMs: 1_000,
    })).resolves.toMatchObject({ allowed: false, available: false });
  });

  it("uses an atomic Redis script when production storage is configured", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("RATE_LIMIT_HASH_SECRET", "independent-rate-limit-secret");
    vi.stubEnv("UPSTASH_REDIS_REST_URL", "https://redis.example.com/");
    vi.stubEnv("UPSTASH_REDIS_REST_TOKEN", "test-token");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ result: [1, 60_000] }), {
        headers: { "Content-Type": "application/json" },
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(checkRateLimit({
      identifier: "203.0.113.12",
      limit: 2,
      namespace: "redis-test",
      windowMs: 60_000,
    })).resolves.toMatchObject({
      allowed: true,
      available: true,
      remaining: 1,
      retryAfterSeconds: 60,
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://redis.example.com");
    expect(init.headers).toMatchObject({ Authorization: "Bearer test-token" });
    const command = JSON.parse(String(init.body)) as string[];
    expect(command[0]).toBe("EVAL");
    expect(command[3]).toMatch(/^audio-transcription:redis-test:/);
    expect(command[3]).not.toContain("203.0.113.12");
  });
});
