import { type FormEvent, useState } from "react";

import { ApiError, api } from "../../lib/api";
import { DEMO_KEY } from "../../lib/demo";
import { Panel } from "../Panel";

type AccessPanelProps = {
  onConnect: (apiKey: string, keepForTab: boolean) => void;
};

function explain(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return "The key could not be checked";
  }
  if (error.status === 401) {
    return "That key was not accepted.";
  }
  if (error.status === 503) {
    return "The API has no key configured. Set MLFLOWHUB_API_KEY and restart it.";
  }
  if (error.status === 0) {
    return "The API is not reachable. Is it running?";
  }
  return error.message;
}

/** Panel 00: enter the API key. It is checked against the API before it is kept. */
export function AccessPanel({ onConnect }: AccessPanelProps) {
  const [value, setValue] = useState("");
  const [keepForTab, setKeepForTab] = useState(false);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const demoKey = DEMO_KEY;

  const connect = async (candidate: string) => {
    if (!candidate) {
      setError("Enter the API key first.");
      return;
    }

    setChecking(true);
    setError(null);
    try {
      await api.verifyKey(candidate);
      onConnect(candidate, keepForTab);
    } catch (caught) {
      setError(explain(caught));
    } finally {
      setChecking(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void connect(value.trim());
  };

  return (
    <Panel index="00" name="Access" state="Locked" className="panel--access">
      <form className="access" onSubmit={submit}>
        <label className="field">
          <span className="label">API key</span>
          <input
            className="field__input mono"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            aria-describedby="access-note"
          />
        </label>

        <label className="check label">
          <input type="checkbox" checked={keepForTab} onChange={(event) => setKeepForTab(event.target.checked)} />
          Keep for this tab
        </label>

        <button className="button button--primary label" type="submit" disabled={checking} data-live={checking}>
          {checking ? "Checking" : "Connect"}
        </button>
      </form>

      {demoKey && (
        <div className="demo-key">
          <p className="label">
            Public demo key <span className="mono demo-key__value">{demoKey}</span>
          </p>
          <button
            type="button"
            className="button button--quiet label"
            disabled={checking}
            onClick={() => {
              setValue(demoKey);
              void connect(demoKey);
            }}
          >
            Use demo key
          </button>
        </div>
      )}

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      <p id="access-note" className="panel__detail panel__fine">
        The key stays in this page and is sent only to this API, in the <span className="mono">X-API-Key</span>{" "}
        header. It is the value of <span className="mono">MLFLOWHUB_API_KEY</span> on the server.
      </p>
    </Panel>
  );
}
