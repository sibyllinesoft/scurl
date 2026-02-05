/**
 * scurl API Worker - Cloudflare Worker that proxies requests to scurl container.
 */

import { getContainer } from "@cloudflare/containers";
import { Env, ScurlContainer } from "./container";
import { checkRateLimit, rateLimitHeaders } from "./ratelimit";

export { ScurlContainer };

interface FetchRequest {
  url: string;
  render?: boolean;
}

interface ConvertRequest {
  html: string;
}

const MAX_CONVERT_SIZE = 256 * 1024; // 256 KB

interface InjectionAnalysis {
  score: number;
  flagged: boolean;
  threshold: number;
  action_taken: string;
  signals: string[];
}

interface ContainerResponse {
  markdown: string | null;
  error: string | null;
  injection: InjectionAnalysis | null;
}

function jsonResponse(data: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      ...headers,
    },
  });
}

function isValidUrl(str: string): boolean {
  try {
    const url = new URL(str);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

async function handleFetch(request: Request, env: Env): Promise<Response> {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";

  // Check rate limit
  const rateLimit = await checkRateLimit(env.RATE_LIMIT, ip);
  const rlHeaders = rateLimitHeaders(rateLimit);

  if (!rateLimit.allowed) {
    return jsonResponse(
      { error: "Rate limit exceeded", retryAfter: rateLimit.resetAt - Math.floor(Date.now() / 1000) },
      429,
      { ...rlHeaders, "Retry-After": String(rateLimit.resetAt - Math.floor(Date.now() / 1000)) }
    );
  }

  // Parse request body
  let body: FetchRequest;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400, rlHeaders);
  }

  if (!body.url || typeof body.url !== "string") {
    return jsonResponse({ error: "Missing 'url' field" }, 400, rlHeaders);
  }

  if (!isValidUrl(body.url)) {
    return jsonResponse({ error: "Invalid URL - must be http or https" }, 400, rlHeaders);
  }

  const container = getContainer(env.SCURL_CONTAINER);

  try {
    const containerResponse = await container.fetch("http://container/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: body.url, render: body.render !== false }),
    });

    const result: ContainerResponse = await containerResponse.json();

    if (result.error) {
      return jsonResponse(
        { error: result.error, url: body.url },
        containerResponse.status >= 500 ? 502 : 400,
        rlHeaders
      );
    }

    return jsonResponse(
      { markdown: result.markdown, url: body.url, cached: false, injection: result.injection },
      200,
      rlHeaders
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return jsonResponse({ error: `Container error: ${message}` }, 502, rlHeaders);
  }
}

async function handleConvert(request: Request, env: Env): Promise<Response> {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";

  const rateLimit = await checkRateLimit(env.RATE_LIMIT, ip);
  const rlHeaders = rateLimitHeaders(rateLimit);

  if (!rateLimit.allowed) {
    return jsonResponse(
      { error: "Rate limit exceeded", retryAfter: rateLimit.resetAt - Math.floor(Date.now() / 1000) },
      429,
      { ...rlHeaders, "Retry-After": String(rateLimit.resetAt - Math.floor(Date.now() / 1000)) }
    );
  }

  let body: ConvertRequest;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400, rlHeaders);
  }

  const text = body.html || (body as any).text;
  if (!text || typeof text !== "string") {
    return jsonResponse({ error: "Missing 'html' or 'text' field" }, 400, rlHeaders);
  }

  const textBytes = new TextEncoder().encode(text).length;
  if (textBytes > MAX_CONVERT_SIZE) {
    return jsonResponse(
      { error: `Input too large (${Math.round(textBytes / 1024)} KB). Maximum is 256 KB.` },
      413,
      rlHeaders
    );
  }

  const container = getContainer(env.SCURL_CONTAINER);

  try {
    const containerResponse = await container.fetch("http://container/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const result: ContainerResponse = await containerResponse.json();

    if (result.error) {
      return jsonResponse(
        { error: result.error },
        containerResponse.status >= 500 ? 502 : 400,
        rlHeaders
      );
    }

    return jsonResponse({ markdown: result.markdown, injection: result.injection }, 200, rlHeaders);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return jsonResponse({ error: `Container error: ${message}` }, 502, rlHeaders);
  }
}

async function handleHealth(): Promise<Response> {
  return jsonResponse({ status: "ok" });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }

    // API routes
    if (url.pathname === "/api/fetch" && request.method === "POST") {
      return handleFetch(request, env);
    }

    if (url.pathname === "/api/convert" && request.method === "POST") {
      return handleConvert(request, env);
    }

    if (url.pathname === "/api/health" && request.method === "GET") {
      return handleHealth();
    }

    // Static assets are served automatically by [assets] config
    return new Response("Not Found", { status: 404 });
  },
};
