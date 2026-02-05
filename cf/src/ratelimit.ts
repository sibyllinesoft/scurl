/**
 * IP-based rate limiting using Cloudflare KV.
 * Allows 10 requests per minute per IP.
 */

const WINDOW_SIZE_SECONDS = 60;
const MAX_REQUESTS = 10;

interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
}

export async function checkRateLimit(
  kv: KVNamespace,
  ip: string
): Promise<RateLimitResult> {
  const now = Math.floor(Date.now() / 1000);
  const windowStart = now - (now % WINDOW_SIZE_SECONDS);
  const key = `ratelimit:${ip}:${windowStart}`;

  const currentCount = parseInt((await kv.get(key)) || "0", 10);

  if (currentCount >= MAX_REQUESTS) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: windowStart + WINDOW_SIZE_SECONDS,
    };
  }

  await kv.put(key, String(currentCount + 1), {
    expirationTtl: WINDOW_SIZE_SECONDS * 2,
  });

  return {
    allowed: true,
    remaining: MAX_REQUESTS - currentCount - 1,
    resetAt: windowStart + WINDOW_SIZE_SECONDS,
  };
}

export function rateLimitHeaders(result: RateLimitResult): Record<string, string> {
  return {
    "X-RateLimit-Limit": String(MAX_REQUESTS),
    "X-RateLimit-Remaining": String(result.remaining),
    "X-RateLimit-Reset": String(result.resetAt),
  };
}
