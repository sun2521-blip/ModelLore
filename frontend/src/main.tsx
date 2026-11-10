import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/archivo/wdth-italic.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";

import { App } from "./App";

const root = document.getElementById("root");

if (!root) {
  throw new Error("ModelLore console: missing #root element");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
