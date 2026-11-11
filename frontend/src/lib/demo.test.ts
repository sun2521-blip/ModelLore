import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchLimits } from "./api";
import { DEMO_KEY, readDemoKey } from "./demo";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("readDemoKey", () => {
  it("is off unless the build sets a key", () => {
    expect(readDemoKey(undefined)).toBeNull();
    expect(readDemoKey("")).toBeNull();
    expect(readDemoKey("   ")).toBeNull();
    expect(readDemoKey(" demo-key ")).toBe("demo-key");
  });

  it("is off in this build, so local and Docker consoles never show a key", () => {
    expect(DEMO_KEY).toBeNull();
  });
});

describe("fetchLimits", () => {
  it("reads the public limits without an API key", async () => {
    const limits = {
      max_upload_mb: 1,
      max_upload_rows: 5000,
      max_upload_columns: 50,
      max_predict_rows: 100,
      rate_limits_per_minute: { upload: 10, train: 6, predict: 60 },
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(limits), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    expect(await fetchLimits()).toEqual(limits);
    expect(fetchMock.mock.calls[0][0]).toBe("/limits");
    expect(fetchMock.mock.calls[0][1]?.headers).toBeUndefined();
  });

  it("gives up quietly when the server is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    expect(await fetchLimits()).toBeNull();
  });
});
