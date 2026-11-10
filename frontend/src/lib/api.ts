/**
 * Thin client for the ModelLore API. The console is served from the same
 * origin as the API (FastAPI in production, Vite's proxy in development), so
 * every path is relative and no CORS configuration is involved.
 */

export type ApiStatus = "checking" | "online" | "offline";

export async function fetchHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/health", { signal });
    return response.ok;
  } catch {
    return false;
  }
}

/** The API's own version, read from its public OpenAPI description. */
export async function fetchApiVersion(signal?: AbortSignal): Promise<string | null> {
  try {
    const response = await fetch("/openapi.json", { signal });
    if (!response.ok) {
      return null;
    }
    const spec: unknown = await response.json();
    const version = (spec as { info?: { version?: unknown } })?.info?.version;
    return typeof version === "string" ? version : null;
  } catch {
    return null;
  }
}
