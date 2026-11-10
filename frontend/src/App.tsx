import { useEffect } from "react";

import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { HudFrame } from "./components/HudFrame";
import { Wordmark } from "./components/Wordmark";
import { useApiStatus } from "./hooks/useApiStatus";

const HERO_NOTES = [
  { position: "nw", title: "Dataset fingerprint", detail: "SHA-256 of every upload" },
  { position: "ne", title: "Run tracking", detail: "Parameters and metrics in MLflow" },
  { position: "sw", title: "Lineage record", detail: "One JSON file per training run" },
  { position: "se", title: "Artifact check", detail: "Hash verified before every load" },
] as const;

const MODULES = [
  {
    index: "01",
    name: "Insert dataset",
    detail: "Upload a CSV. ModelLore checks it, fingerprints it, and keeps it for training.",
    endpoint: "POST /datasets/upload",
  },
  {
    index: "02",
    name: "Record",
    detail: "Train on Iris or on your dataset. The counter runs while the model trains.",
    endpoint: "POST /train",
  },
  {
    index: "03",
    name: "Playback",
    detail: "Every run becomes a bar. The playhead picks one and reads out its metrics.",
    endpoint: "GET /jobs · /models · /runs",
  },
  {
    index: "04",
    name: "Trace",
    detail: "Take a model apart: dataset, configuration, run, and artifact, with their fingerprints.",
    endpoint: "GET /lineage",
  },
  {
    index: "05",
    name: "Play",
    detail: "Send feature rows to a trained model and read back its predictions.",
    endpoint: "POST /predict",
  },
];

export function App() {
  const { status, version } = useApiStatus();

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

          {HERO_NOTES.map((note) => (
            <p key={note.position} className={`hero__note hero__note--${note.position} label`}>
              <span className="hero__note-tick" aria-hidden="true" />
              {note.title}
              <br />
              <span className="mono hero__note-detail">{note.detail}</span>
            </p>
          ))}
        </section>

        <section className="console" id="console" aria-labelledby="console-title">
          <div className="section-rule">
            <span className="section-rule__flare" aria-hidden="true" />
            <h2 id="console-title" className="label">
              Console
            </h2>
            <span className="section-rule__flare section-rule__flare--end" aria-hidden="true" />
          </div>

          <div className="module-grid">
            {MODULES.map((module) => (
              <article key={module.index} className="panel">
                <header className="panel__head">
                  <span className="mono panel__index">{module.index}</span>
                  <h3 className="label panel__name">{module.name}</h3>
                  <span className="label panel__state">Next step</span>
                </header>
                <p className="panel__detail">{module.detail}</p>
                <p className="mono panel__endpoint">{module.endpoint}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <Footer version={version} />
    </div>
  );
}
