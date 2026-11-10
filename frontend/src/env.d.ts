/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set only for the hosted public demo: a shared key the console offers to visitors. */
  readonly VITE_DEMO_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
