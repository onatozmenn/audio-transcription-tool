import { createHmac } from "node:crypto";

type RateLimitOptions = {
  identifier: string;
  limit: number;
  namespace: string;
  windowMs: number;
};

export type RateLimitResult = {
  allowed: boolean;
  available: boolean;
  remaining: number;
  retryAfterSeconds: number;
};

const LOCAL_RATE_LIMIT_STORE = new Map<string, { count: number; expiresAt: number }>();
let warnedAboutMissingStore = false;
const REDIS_SCRIPT = [
  "local count = redis.call('INCR', KEYS[1])",
  "if count == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end",
  "local ttl = redis.call('PTTL', KEYS[1])",
  "return {count, ttl}",
].join("\n");

function hashIdentifier(identifier: string): string {
  const secret =
    process.env.RATE_LIMIT_HASH_SECRET ??
    process.env.TRANSCRIPTION_SESSION_SECRET ??
    process.env.GROQ_API_KEY ??
    "local-development-only";

  return createHmac("sha256", secret)
    .update(identifier)
    .digest("base64url")
    .slice(0, 32);
}

function localRateLimit(options: RateLimitOptions, key: string): RateLimitResult {
  const now = Date.now();
  const current = LOCAL_RATE_LIMIT_STORE.get(key);
  const state = !current || current.expiresAt <= now
    ? { count: 1, expiresAt: now + options.windowMs }
    : { ...current, count: current.count + 1 };

  LOCAL_RATE_LIMIT_STORE.set(key, state);

  if (LOCAL_RATE_LIMIT_STORE.size > 1_000) {
    for (const [storedKey, storedState] of LOCAL_RATE_LIMIT_STORE) {
      if (storedState.expiresAt <= now) LOCAL_RATE_LIMIT_STORE.delete(storedKey);
    }
  }

  return {
    allowed: state.count <= options.limit,
    available: true,
    remaining: Math.max(options.limit - state.count, 0),
    retryAfterSeconds: Math.max(1, Math.ceil((state.expiresAt - now) / 1_000)),
  };
}

function unavailableRateLimit(): RateLimitResult {
  return {
    allowed: false,
    available: false,
    remaining: 0,
    retryAfterSeconds: 60,
  };
}

export async function checkRateLimit(options: RateLimitOptions): Promise<RateLimitResult> {
  const identifierHash = hashIdentifier(options.identifier);
  const key = `audio-transcription:${options.namespace}:${identifierHash}`;
  const redisUrl = (
    process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL
  )?.replace(/\/$/, "");
  const redisToken = process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;

  // No shared store configured. Degrade to the in-process limiter instead of
  // rejecting every request: a missing store is a deployment choice, and
  // failing closed here takes the whole cloud transcription path offline.
  // Per-instance counters are weaker than distributed ones, but they still
  // throttle the abusive bursts this guard exists to stop.
  if (!redisUrl || !redisToken) {
    if (process.env.NODE_ENV === "production" && !warnedAboutMissingStore) {
      warnedAboutMissingStore = true;
      console.warn(
        "Rate limiting is running per-instance: set UPSTASH_REDIS_REST_URL/TOKEN " +
        "(or KV_REST_API_URL/TOKEN) to enable distributed limits.",
      );
    }

    return localRateLimit(options, key);
  }

  try {
    const response = await fetch(redisUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${redisToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify([
        "EVAL",
        REDIS_SCRIPT,
        "1",
        key,
        String(options.windowMs),
      ]),
      cache: "no-store",
      signal: AbortSignal.timeout(3_000),
    });

    if (!response.ok) return unavailableRateLimit();

    const payload = (await response.json()) as { result?: unknown };
    if (!Array.isArray(payload.result) || payload.result.length < 2) {
      return unavailableRateLimit();
    }

    const count = Number(payload.result[0]);
    const ttlMs = Number(payload.result[1]);
    if (!Number.isFinite(count) || !Number.isFinite(ttlMs)) {
      return unavailableRateLimit();
    }

    return {
      allowed: count <= options.limit,
      available: true,
      remaining: Math.max(options.limit - count, 0),
      retryAfterSeconds: Math.max(1, Math.ceil(ttlMs / 1_000)),
    };
  } catch (error) {
    console.error("Distributed rate limit unavailable:", error);
    return unavailableRateLimit();
  }
}