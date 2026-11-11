import { type ChangeEvent, type DragEvent, useEffect, useState } from "react";

import { ApiError, api, fetchLimits, type Limits, type UploadResult } from "../../lib/api";
import { readCsvHeader } from "../../lib/csv";
import { shortHash } from "../../lib/format";
import { Panel } from "../Panel";

type UploadPanelProps = {
  apiKey: string;
  onUploaded: (result: UploadResult, columns: string[]) => void;
};

/** Panel 01: insert a CSV. The server validates and fingerprints it. */
export function UploadPanel({ apiKey, onUploaded }: UploadPanelProps) {
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [limits, setLimits] = useState<Limits | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchLimits(controller.signal).then((value) => {
      if (!controller.signal.aborted) {
        setLimits(value);
      }
    });
    return () => controller.abort();
  }, []);

  const upload = async (file: File | undefined) => {
    if (!file) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const [columns, uploaded] = await Promise.all([readCsvHeader(file), api.uploadDataset(apiKey, file)]);
      setResult(uploaded);
      onUploaded(uploaded, columns);
    } catch (caught) {
      setResult(null);
      setError(caught instanceof ApiError ? caught.message : "The file could not be uploaded");
    } finally {
      setBusy(false);
    }
  };

  const onChoose = (event: ChangeEvent<HTMLInputElement>) => {
    void upload(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    void upload(event.dataTransfer.files?.[0]);
  };

  return (
    <Panel index="01" name="Insert dataset" state={busy ? "Reading" : result ? "Inserted" : "Empty"}>
      <label
        className="slot"
        data-dragging={dragging}
        data-busy={busy}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input className="visually-hidden" type="file" accept=".csv,text/csv" onChange={onChoose} disabled={busy} />
        <svg className="slot__window" viewBox="0 0 150 50" aria-hidden="true">
          <path
            d="M13 1.5 H137 L148.5 13 V37 L137 48.5 H13 L1.5 37 V13 Z"
            fill="none"
            stroke="var(--ink)"
            strokeWidth={2}
            strokeLinejoin="round"
          />
          {[46, 104].map((cx) => (
            <g key={cx} className="slot__reel">
              <circle cx={cx} cy={25} r={13} fill="none" stroke="var(--ink)" strokeWidth={3} />
              <circle cx={cx} cy={25} r={3} fill="var(--ink)" />
              {[0, 120, 240].map((angle) => (
                <line
                  key={angle}
                  x1={cx}
                  y1={25}
                  x2={cx + 11 * Math.cos((angle * Math.PI) / 180)}
                  y2={25 + 11 * Math.sin((angle * Math.PI) / 180)}
                  stroke="var(--ink)"
                  strokeWidth={3}
                  strokeLinecap="round"
                />
              ))}
            </g>
          ))}
          <line x1={59} y1={25} x2={91} y2={25} stroke="var(--ink)" strokeWidth={1.5} strokeDasharray="3 3" />
        </svg>
        <span className="label">{busy ? "Checking the file" : "Drop a CSV here, or choose a file"}</span>
        <span className="mono slot__hint">
          {limits
            ? `Up to ${limits.max_upload_mb} MiB · ${limits.max_upload_rows.toLocaleString("en")} rows · ${limits.max_upload_columns} columns`
            : "Size, row and column limits are set by the server"}
        </span>
      </label>

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <dl className="readout">
          <div>
            <dt className="label">Rows</dt>
            <dd className="mono">{result.row_count}</dd>
          </div>
          <div>
            <dt className="label">Columns</dt>
            <dd className="mono">{result.column_count}</dd>
          </div>
          <div>
            <dt className="label">Fingerprint</dt>
            <dd className="mono" title={result.sha256}>
              {shortHash(result.sha256)}
            </dd>
          </div>
          <div>
            <dt className="label">Dataset</dt>
            <dd className="mono" title={result.dataset_id}>
              {shortHash(result.dataset_id.replace(/-/g, ""), 8, 4)}
            </dd>
          </div>
        </dl>
      )}
    </Panel>
  );
}
