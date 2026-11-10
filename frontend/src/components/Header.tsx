import type { ApiStatus } from "../lib/api";
import { Wordmark } from "./Wordmark";

const STATUS_TEXT: Record<ApiStatus, string> = {
  checking: "CHECKING",
  online: "ONLINE",
  offline: "OFFLINE",
};

type HeaderProps = {
  status: ApiStatus;
  version: string | null;
};

export function Header({ status, version }: HeaderProps) {
  return (
    <header className="topbar">
      <div className="topbar__side">
        <p className="label">
          Lineage recorder
          <br />
          <span className="mono">ML-L {version ?? "-.-.-"}</span>
        </p>
        <p className="label topbar__rec">
          <span className="rec-dot" data-live={status === "online"} />
          Train · Track · Trace
        </p>
      </div>

      <a className="topbar__tab" href="#top" aria-label="ModelLore, back to the top">
        <svg className="topbar__tab-shape" viewBox="0 0 240 52" preserveAspectRatio="none" aria-hidden="true">
          <path
            d="M0 0.75 L18 44 Q21 51 29 51 L211 51 Q219 51 222 44 L240 0.75"
            fill="var(--sage)"
            stroke="var(--ink)"
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
        <Wordmark variant="tab" />
      </a>

      <div className="topbar__side topbar__side--right">
        <p className="label" aria-live="polite">
          API <span className="mono" data-status={status}>{STATUS_TEXT[status]}</span>
        </p>
        <a className="label topbar__link" href="#console">
          Open console
        </a>
      </div>
    </header>
  );
}
