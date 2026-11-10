import { type ChangeEvent, type DragEvent, useState } from "react";

import { ApiError, api, type UploadResult } from "../../lib/api";
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
        <span className="slot__window" aria-hidden="true">
          <span className="slot__reel" />
          <span className="slot__reel" />
        </span>
        <span className="label">{busy ? "Checking the file" : "Drop a CSV here, or choose a file"}</span>
        <span className="mono slot__hint">Up to 10 MiB · 100,000 rows · 200 columns</span>
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
