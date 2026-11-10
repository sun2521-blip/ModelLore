import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite runs this file in Node; declared here so the editor needs no @types/node.
declare const process: { env: Record<string, string | undefined> };

// During development the console runs on Vite's server and forwards every API
// call to the FastAPI app, so the browser only ever talks to one origin and the
// API needs no CORS configuration. Point it elsewhere with MODELLORE_API_URL,
// for example when port 8000 is already taken by a Docker stack.
const API_TARGET = process.env.MODELLORE_API_URL ?? "http://127.0.0.1:8000";
const API_PATHS = [
  "/health",
  "/datasets",
  "/train",
  "/jobs",
  "/models",
  "/runs",
  "/predict",
  "/lineage",
  "/openapi.json",
];

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(API_PATHS.map((path) => [path, API_TARGET])),
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
