import { describe, expect, it } from "vitest";

import { parseCsvHeader, readCsvHeader } from "./csv";

describe("parseCsvHeader", () => {
  it("reads a plain header row", () => {
    expect(parseCsvHeader("sepal_length,sepal_width,target\n5.1,3.5,setosa\n")).toEqual([
      "sepal_length",
      "sepal_width",
      "target",
    ]);
  });

  it("handles quotes, embedded commas, escaped quotes, and CRLF", () => {
    expect(parseCsvHeader('"a, b","say ""hi""",c\r\n1,2,3')).toEqual(["a, b", 'say "hi"', "c"]);
  });

  it("skips a byte-order mark and blank lines before the header", () => {
    expect(parseCsvHeader("﻿\n\nfeature,target\n1,0")).toEqual(["feature", "target"]);
  });

  it("returns nothing for an empty file", () => {
    expect(parseCsvHeader("")).toEqual([]);
  });
});

describe("readCsvHeader", () => {
  it("only needs the start of the file", async () => {
    const file = new Blob(["feature,target\n", "1,0\n".repeat(50_000)]);
    expect(await readCsvHeader(file, 64)).toEqual(["feature", "target"]);
  });
});
