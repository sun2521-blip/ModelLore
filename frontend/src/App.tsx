import { useEffect } from "react";

import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { HudFrame } from "./components/HudFrame";
import { Finale, Marquee } from "./components/Story";
import { Wordmark } from "./components/Wordmark";
import { Console } from "./components/console/Console";
import { useApiKey } from "./hooks/useApiKey";
import { useApiStatus } from "./hooks/useApiStatus";
import { useConsoleData } from "./hooks/useConsoleData";

const HERO_NOTES = [
  { position: "nw", title: "Dataset fingerprint", detail: "SHA-256 of every upload" },
  { position: "ne", title: "Run tracking", detail: "Parameters and metrics in MLflow" },
  { position: "sw", title: "Lineage record", detail: "One JSON file per training run" },
  { position: "se", title: "Artifact check", detail: "Hash verified before every load" },
] as const;

export function App() {
  const { status, version } = useApiStatus();
  const { apiKey, setApiKey } = useApiKey();
  const data = useConsoleData(apiKey);

  // The browser looks for a #section before React has drawn it, so honour
  // links like /#console once the page exists.
  useEffect(() => {
    const sectionId = window.location.hash.slice(1);
    if (sectionId) {
      document.getElementById(sectionId)?.scrollIntoView();
    }
  }, []);

  return (
    <div className="page">
      <HudFrame />
      <Header status={status} version={version} />

      <main>
        <section className="hero" id="top" aria-labelledby="hero-title">
          <h1 id="hero-title" className="visually-hidden">
            ModelLore
          </h1>
          <div className="hero__mark">
            <Wordmark variant="hero" />
          </div>
          <p className="hero__tagline label">Know how your models came to be.</p>
          <p className="hero__lede">
            A configuration-driven training and lineage platform for tabular classification. Upload a
            dataset, train a model, and trace it back to the data and configuration that made it.
          </p>
          <a className="button button--primary label hero__cta" href="#console">
            Open the console
          </a>

          {HERO_NOTES.map((note) => (
            <p key={note.position} className={`hero__note hero__note--${note.position} label`}>
              <span className="hero__note-tick" aria-hidden="true" />
              {note.title}
              <br />
              <span className="mono hero__note-detail">{note.detail}</span>
            </p>
          ))}
        </section>

        <Marquee />

        <section className="console" id="console" aria-labelledby="console-title">
          <div className="section-rule">
            <span className="section-rule__flare" aria-hidden="true" />
            <h2 id="console-title" className="label">
              Console
            </h2>
            <span className="section-rule__flare section-rule__flare--end" aria-hidden="true" />
          </div>

          <Console status={status} apiKey={apiKey} onKey={setApiKey} data={data} />
        </section>

        <Finale />
      </main>

      <Footer version={version} runCount={apiKey ? data.jobs.length : null} />
    </div>
  );
}
