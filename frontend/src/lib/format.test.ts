import { describe, expect, it } from "vitest";

import { percent, serial, shortHash, tapeCounter } from "./format";

describe("shortHash", () => {
  it("keeps the head and tail of a long fingerprint", () => {
    expect(shortHash("aac7fc942867767ebe1bbe8f616e1314900dbf35389f6e245e16850ba7d87f3f")).toBe(
      "aac7fc94…7f3f",
    );
  });

  it("leaves short values alone and marks missing ones", () => {
    expect(shortHash("abc123")).toBe("abc123");
    expect(shortHash(null)).toBe("—");
    expect(shortHash("")).toBe("—");
  });
});

describe("tapeCounter", () => {
  it("formats elapsed time as hours, minutes and seconds", () => {
    expect(tapeCounter(0)).toBe("00:00:00");
    expect(tapeCounter(9_336)).toBe("00:00:09");
    expect(tapeCounter(202_000)).toBe("00:03:22");
    expect(tapeCounter(3_723_000)).toBe("01:02:03");
  });

  it("never runs backwards", () => {
    expect(tapeCounter(-500)).toBe("00:00:00");
  });
});

describe("serial", () => {
  it("zero-pads counts", () => {
    expect(serial(7)).toBe("000007");
    expect(serial(3214, 12)).toBe("000000003214");
  });

  it("shows dashes when there is nothing to count yet", () => {
    expect(serial(null)).toBe("------");
    expect(serial(Number.NaN, 3)).toBe("---");
  });
});

describe("percent", () => {
  it("renders metrics with one decimal", () => {
    expect(percent(0.9023569)).toBe("90.2%");
    expect(percent(1)).toBe("100.0%");
    expect(percent(undefined)).toBe("—");
  });
});
