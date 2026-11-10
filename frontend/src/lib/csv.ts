/**
 * Read the header row of a CSV in the browser, so the console can offer the
 * columns as target choices. Only the first record is parsed; nothing else in
 * the file is read or evaluated.
 */
export function parseCsvHeader(text: string): string[] {
  const source = text.replace(/^﻿/, "");
  const columns: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (inQuotes) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        inQuotes = false;
      } else {
        field += character;
      }
      continue;
    }

    if (character === '"') {
      inQuotes = true;
    } else if (character === ",") {
      columns.push(field.trim());
      field = "";
    } else if (character === "\n" || character === "\r") {
      if (columns.length > 0 || field.trim()) {
        break;
      }
      // Skip blank lines before the header.
    } else {
      field += character;
    }
  }

  if (columns.length > 0 || field.trim()) {
    columns.push(field.trim());
  }

  return columns.filter((column) => column.length > 0);
}

/** Read only the start of a file: enough for a header row, never the whole dataset. */
export async function readCsvHeader(file: Blob, maxBytes = 64 * 1024): Promise<string[]> {
  const head = await file.slice(0, maxBytes).text();
  return parseCsvHeader(head);
}
