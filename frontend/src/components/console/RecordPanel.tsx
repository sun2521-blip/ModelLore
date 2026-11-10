import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, api, type Dataset, type TrainResult } from "../../lib/api";
import { percent, shortHash, tapeCounter } from "../../lib/format";
import { Panel } from "../Panel";

type RecordPanelProps = {
  apiKey: string;
  datasets: Dataset[];
  /** Header rows read in the browser for datasets uploaded in this session. */
  knownColumns: Record<string, string[]>;
  preferredDatasetId: string | null;
  onRecorded: (result: TrainResult) => void;
};

function defaultTarget(columns: string[] | undefined, dataset: Dataset | undefined): string {
  if (dataset?.target_column) {
    return dataset.target_column;
  }
  if (columns?.includes("target")) {
    return "target";
  }
  return columns?.[columns.length - 1] ?? "target";
}

/** Panel 02: train a model. The counter runs for real while the API trains. */
export function RecordPanel({ apiKey, datasets, knownColumns, preferredDatasetId, onRecorded }: RecordPanelProps) {
  const [source, setSource] = useState<"iris" | "dataset">("iris");
  const [datasetId, setDatasetId] = useState<string>("");
  const [target, setTarget] = useState("target");
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startedAt = useRef(0);

  const dataset = useMemo(() => datasets.find((item) => item.dataset_id === datasetId), [datasets, datasetId]);
  const columns = datasetId ? knownColumns[datasetId] : undefined;

  // A freshly inserted dataset becomes the one to record.
  useEffect(() => {
    if (preferredDatasetId) {
      setSource("dataset");
      setDatasetId(preferredDatasetId);
    }
  }, [preferredDatasetId]);

  useEffect(() => {
    if (!datasetId && datasets.length > 0) {
      setDatasetId(datasets[0].dataset_id);
    }
  }, [datasets, datasetId]);

  useEffect(() => {
    setTarget(defaultTarget(columns, dataset));
  }, [columns, dataset]);

  useEffect(() => {
    if (!recording) {
      return;
    }
    const timer = window.setInterval(() => setElapsed(performance.now() - startedAt.current), 250);
    return () => window.clearInterval(timer);
  }, [recording]);

  const record = async (event: FormEvent) => {
    event.preventDefault();
    if (source === "dataset" && !datasetId) {
      setError("Insert a dataset first, or record with Iris.");
      return;
    }

    startedAt.current = performance.now();
    setElapsed(0);
    setRecording(true);
    setError(null);
    setResult(null);
    try {
      const trained = await api.train(
        apiKey,
        source === "iris" ? { config_name: "iris" } : { dataset_id: datasetId, target_column: target.trim() },
      );
      setResult(trained);
      onRecorded(trained);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Training could not be started");
    } finally {
      setElapsed(performance.now() - startedAt.current);
      setRecording(false);
    }
  };

  return (
    <Panel index="02" name="Record" state={recording ? "Recording" : result ? "Recorded" : "Standby"}>
      <form className="record" onSubmit={record}>
        <fieldset className="toggle">
          <legend className="visually-hidden">Training data</legend>
          <label className="toggle__option label" data-active={source === "iris"}>
            <input type="radio" name="source" checked={source === "iris"} onChange={() => setSource("iris")} />
            Iris, built in
          </label>
          <label className="toggle__option label" data-active={source === "dataset"} data-disabled={datasets.length === 0}>
            <input
              type="radio"
              name="source"
              checked={source === "dataset"}
              disabled={datasets.length === 0}
              onChange={() => setSource("dataset")}
            />
            Inserted dataset
          </label>
        </fieldset>

        {source === "dataset" && (
          <div className="record__fields">
            <label className="field">
              <span className="label">Dataset</span>
              <select className="field__input" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
                {datasets.map((item) => (
                  <option key={item.dataset_id} value={item.dataset_id}>
                    {item.original_filename} · {item.row_count ?? "?"} rows
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span className="label">Target column</span>
              {columns && columns.length > 0 ? (
                <select className="field__input" value={target} onChange={(event) => setTarget(event.target.value)}>
                  {columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="field__input mono"
                  value={target}
                  maxLength={255}
                  onChange={(event) => setTarget(event.target.value)}
                />
              )}
            </label>
          </div>
        )}

        <div className="record__deck">
          <button
            className="button button--primary record__button label"
            type="submit"
            disabled={recording}
            data-live={recording}
          >
            <span className="rec-dot" data-live={recording} aria-hidden="true" />
            {recording ? "Recording" : "Record"}
          </button>
          <p className="counter mono" aria-live="polite" aria-label={`Elapsed ${tapeCounter(elapsed)}`}>
            {tapeCounter(elapsed)}
          </p>
        </div>
      </form>

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <dl className="readout">
          <div>
            <dt className="label">Accuracy</dt>
            <dd className="mono">{percent(result.metrics.accuracy)}</dd>
          </div>
          <div>
            <dt className="label">F1</dt>
            <dd className="mono">{percent(result.metrics.f1)}</dd>
          </div>
          <div>
            <dt className="label">Lineage run</dt>
            <dd className="mono" title={result.lineage_run_id}>
              {shortHash(result.lineage_run_id.replace(/-/g, ""), 8, 4)}
            </dd>
          </div>
          <div>
            <dt className="label">MLflow run</dt>
            <dd className="mono" title={result.mlflow_run_id}>
              {shortHash(result.mlflow_run_id, 8, 4)}
            </dd>
          </div>
        </dl>
      )}
    </Panel>
  );
}
