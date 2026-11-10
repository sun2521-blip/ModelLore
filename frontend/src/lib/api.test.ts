import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, describeDetail } from "./api";

function respond(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("describeDetail", () => {
  it("uses a plain detail message as-is", () => {
    const error = describeDetail(400, { detail: "Target column 'x' not found in dataset." });
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(400);
    expect(error.message).toBe("Target column 'x' not found in dataset.");
  });

  it("names the field from a validation error", () => {
    const error = describeDetail(422, {
      detail: [{ loc: ["body", "target_column"], msg: "String should have at most 255 characters" }],
    });
    expect(error.message).toBe("target_column: String should have at most 255 characters");
  });

  it("keeps the expected columns from a prediction mismatch", () => {
    const error = describeDetail(422, {
      detail: { message: "Each feature row must contain exactly the feature columns used during training.", expected_columns: ["a", "b"] },
    });
    expect(error.expectedColumns).toEqual(["a", "b"]);
  });

  it("falls back to a readable message when there is no detail", () => {
    expect(describeDetail(429, null).message).toMatch(/too many requests/i);
    expect(describeDetail(401, {}).message).toMatch(/not accepted/i);
    expect(describeDetail(418, {}).message).toBe("The API answered 418");
  });
});

describe("api client", () => {
  it("sends the key in the X-API-Key header and nowhere else", async () => {
    const fetchMock = respond(200, { items: [], total: 0, limit: 100, offset: 0 });
    vi.stubGlobal("fetch", fetchMock);

    await api.listJobs("secret-key");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/jobs?limit=100");
    expect(String(url)).not.toContain("secret-key");
    expect(init.headers).toEqual({ "X-API-Key": "secret-key" });
  });

  it("posts JSON for training and predictions", async () => {
    const fetchMock = respond(200, { lineage_run_id: "run", predictions: [0], count: 1 });
    vi.stubGlobal("fetch", fetchMock);

    await api.predict("k", "run", [{ "sepal length (cm)": 5.1 }]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/predict");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ lineage_run_id: "run", features: [{ "sepal length (cm)": 5.1 }] });
  });

  it("encodes IDs placed in the path", async () => {
    const fetchMock = respond(200, {});
    vi.stubGlobal("fetch", fetchMock);

    await api.getLineage("k", "../../etc/passwd");

    expect(fetchMock.mock.calls[0][0]).toBe("/lineage/..%2F..%2Fetc%2Fpasswd");
  });

  it("turns error answers into ApiError", async () => {
    vi.stubGlobal("fetch", respond(503, { detail: "API authentication is not configured" }));

    await expect(api.verifyKey("k")).rejects.toMatchObject({
      status: 503,
      message: "API authentication is not configured",
    });
  });

  it("reports an unreachable API as status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(api.listModels("k")).rejects.toMatchObject({ status: 0 });
  });
});
