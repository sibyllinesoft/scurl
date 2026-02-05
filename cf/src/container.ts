/**
 * Durable Object class for the scurl container.
 * Uses @cloudflare/containers Container base class.
 */

import { Container } from "@cloudflare/containers";

export interface Env {
  SCURL_CONTAINER: DurableObjectNamespace<ScurlContainer>;
  RATE_LIMIT: KVNamespace;
}

export class ScurlContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";
}
