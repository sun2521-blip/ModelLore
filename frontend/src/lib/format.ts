/** Formatting helpers for the instrument-style readouts. */

/** Shorten a hex fingerprint to its first and last characters: "aac7fc94…5f3f". */
export function shortHash(hash: string | null | undefined, head = 8, tail = 4): string {
  if (!hash) {
    return "—";
  }
  if (hash.length <= head + tail + 1) {
    return hash;
  }
  return `${hash.slice(0, head)}…${hash.slice(-tail)}`;
}

/** Render milliseconds as a tape counter: 00:03:22. */
export function tapeCounter(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((part) => String(part).padStart(2, "0")).join(":");
}

/** Zero-pad a count like a serial number: 7 becomes "000007". */
export function serial(value: number | null | undefined, width = 6): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-".repeat(width);
  }
  return String(Math.max(0, Math.trunc(value))).padStart(width, "0");
}

/** A metric between 0 and 1 as a percentage with one decimal: 0.9023 becomes "90.2%". */
export function percent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(1)}%`;
}
