import { type FormEvent, useEffect, useState } from "react";

import { ApiError, api, type Job, type Model, type PredictResult } from "../../lib/api";
import { describePrediction, exampleFeatures, isPlayable, parseFeatureValue } from "../../lib/runs";
import { Panel } from "../Panel";

type PlayPanelProps = {
  apiKey: string;
  job: Job | undefined;
  model: Model | undefined;
};

const MAX_ROWS = 10;

/** Panel 05: send feature rows to the selected model and read back its predictions. */
export function PlayPanel({ apiKey, job, model }: PlayPanelProps) {
  const [rows, setRows] = useState<Array<Record<string, string>>>([]);
  const [playing, setPlaying] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const playable = isPlayable(job, model);
  const columns = model?.feature_columns ?? [];

  useEffect(() => {
    setRows(model ? [exampleFeatures(model)] : []);
    setResult(null);
    setError(null);
  }, [model]);

  const update = (rowIndex: number, column: string, value: string) => {
    setRows((current) => current.map((row, index) => (index === rowIndex ? { ...row, [column]: value } : row)));
  };

  const play = async (event: FormEvent) => {
    event.preventDefault();
    if (!model?.lineage_run_id) {
      return;
    }
    setPlaying(true);
    setError(null);
    try {
      const features = rows.map((row) =>
        Object.fromEntries(columns.map((column) => [column, parseFeatureValue(row[column] ?? "")])),
      );
      setResult(await api.predict(apiKey, model.lineage_run_id, features));
    } catch (caught) {
      setResult(null);
      if (caught instanceof ApiError && caught.expectedColumns) {
        setError(`${caught.message} Expected: ${caught.expectedColumns.join(", ")}`);
      } else {
        setError(caught instanceof ApiError ? caught.message : "The prediction could not be made");
      }
    } finally {
      setPlaying(false);
    }
  };

  return (
    <Panel index="05" name="Play" state={playing ? "Playing" : result ? "Played" : playable ? "Ready" : "No model"}
      id="play"
      className="panel--wide"
    >
      {!playable && (
        <p className="panel__detail">
          {job ? "This run has no model to play. Pick a finished run in playback." : "Pick a finished run in playback."}
        </p>
      )}

      {playable && (
        <form className="play" onSubmit={play}>
          <div className="play__rows">
            {rows.map((row, rowIndex) => (
              <fieldset key={rowIndex} className="play__row">
                <legend className="label">Row {rowIndex + 1}</legend>
                {columns.map((column) => (
                  <label key={column} className="field field--compact">
                    <span className="label">{column}</span>
                    <input
                      className="field__input mono"
                      value={row[column] ?? ""}
                      maxLength={1000}
                      onChange={(event) => update(rowIndex, column, event.target.value)}
                    />
                  </label>
                ))}
                {rows.length > 1 && (
                  <button
                    type="button"
                    className="button button--quiet label"
                    onClick={() => setRows((current) => current.filter((_, index) => index !== rowIndex))}
                  >
                    Remove row
                  </button>
                )}
              </fieldset>
            ))}
          </div>

          <div className="play__deck">
            <button
              type="button"
              className="button button--quiet label"
              disabled={rows.length >= MAX_ROWS}
              onClick={() => setRows((current) => [...current, exampleFeatures(model)])}
            >
              Add row
            </button>
            <button className="button button--primary label" type="submit" disabled={playing} data-live={playing}>
              <span className="play__triangle" aria-hidden="true" />
              {playing ? "Playing" : "Play"}
            </button>
          </div>
        </form>
      )}

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <ol className="predictions" aria-label="Predictions">
          {result.predictions.map((value, index) => (
            <li key={index}>
              <span className="label">Row {index + 1}</span>
              <span className="display predictions__value">{describePrediction(model, value)}</span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
