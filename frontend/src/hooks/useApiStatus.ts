import { useEffect, useState } from "react";

import { type ApiStatus, fetchApiVersion, fetchHealth } from "../lib/api";

const POLL_MS = 15_000;

/** Poll the public health check, and read the API version the first time it answers. */
export function useApiStatus() {
  const [status, setStatus] = useState<ApiStatus>("checking");
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let versionKnown = false;

    const check = async () => {
      const healthy = await fetchHealth(controller.signal);
      if (controller.signal.aborted) {
        return;
      }
      setStatus(healthy ? "online" : "offline");

      if (healthy && !versionKnown) {
        const apiVersion = await fetchApiVersion(controller.signal);
        if (!controller.signal.aborted && apiVersion) {
          versionKnown = true;
          setVersion(apiVersion);
        }
      }
      timer = window.setTimeout(check, POLL_MS);
    };

    void check();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  return { status, version };
}
