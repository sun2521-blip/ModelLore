/**
 * The hosted public demo. When the build sets VITE_DEMO_KEY, the console
 * offers that shared key to visitors and says plainly what a shared,
 * self-resetting demo means. Local and Docker builds leave it unset.
 */
export function readDemoKey(value: string | undefined): string | null {
  const key = value?.trim();
  return key ? key : null;
}

export const DEMO_KEY = readDemoKey(import.meta.env.VITE_DEMO_KEY);

export const DEMO_NOTICE =
  "This is a shared public demo. Other visitors can see what you upload, and everything resets " +
  "when the server sleeps after 15 quiet minutes. Please don't upload anything private.";

export const DEMO_WAKING =
  "The demo server is asleep or waking up. The first answer can take about a minute; this page " +
  "reconnects on its own.";
