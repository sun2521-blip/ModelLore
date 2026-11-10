import { serial } from "../lib/format";

type FooterProps = {
  version: string | null;
};

export function Footer({ version }: FooterProps) {
  return (
    <footer className="bottombar">
      <div className="bottombar__spec label">
        <span className="bottombar__badge mono">ML</span>
        <span>
          ModelLore · v{version ?? "-.-.-"} · MIT license
          <span className="bottombar__tagline">
            <br />
            Train it. Track it. Trace it.
          </span>
        </span>
      </div>

      <a className="bottombar__tab label" href="#console">
        <svg className="bottombar__tab-shape" viewBox="0 0 240 48" preserveAspectRatio="none" aria-hidden="true">
          <path
            d="M0 47.25 L16 7 Q19 0.75 27 0.75 L213 0.75 Q221 0.75 224 7 L240 47.25"
            fill="var(--signal)"
            stroke="var(--ink)"
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        <span>Trace a model</span>
      </a>

      <p className="bottombar__serial label mono">{serial(null, 12)}</p>
    </footer>
  );
}
