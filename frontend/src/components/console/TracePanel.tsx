import { type ReactNode, useEffect, useState } from "react";

import { ApiError, api, type Job, type Lineage } from "../../lib/api";
import { percent, shortHash } from "../../lib/format";
import { useInView } from "../../hooks/useInView";
import { Panel } from "../Panel";
import { LineageDrawing } from "./LineageDrawing";

type TracePanelProps = {
  apiKey: string;
  job: Job | undefined;
};

function Hash({ value, head = 10, tail = 6 }: { value: string | null | undefined; head?: number; tail?: number }) {
  return (
    <span className="mono" title={value ?? undefined}>
      {shortHash(value, head, tail)}
    </span>
  );
}

function SpecBlock({ title, rows }: { title: string; rows: Array<[string, ReactNode]> }) {
  return (
    <div className="spec-block">
      <h4 className="label spec-block__title">{title}</h4>
      <dl className="spec-block__rows">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt className="label">{label}</dt>
            <dd>{value ?? "—"}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Panel 04: take the selected model apart, from its artifact down to its data. */
export function TracePanel({ apiKey, job }: TracePanelProps) {
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { ref, inView } = useInView<HTMLDivElement>(0.3);
  const lineageRunId = job?.lineage_run_id ?? null;

  useEffect(() => {
    setLineage(null);
    setError(null);
    if (!lineageRunId) {
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    api
      .getLineage(apiKey, lineageRunId, controller.signal)
      .then(setLineage)
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof ApiError ? caught.message : "The lineage record could not be read");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [apiKey, lineageRunId]);

  const state = !job ? "No run" : !lineageRunId ? "Nothing to trace" : loading ? "Reading" : lineage ? "Traced" : "—";
  const environment = lineage?.model.environment;
  const parameters = lineage?.config.parameters;

  return (
    <Panel index="04" name="Trace" state={state} className="panel--wide" id="trace">
      {!job && <p className="panel__detail">Pick a run in playback to trace the model it produced.</p>}
      {job && !lineageRunId && (
        <p className="panel__detail">
          This run left no model behind, so there is nothing to take apart. Its error is shown in playback.
        </p>
      )}
      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {lineageRunId && (
        <div className="trace" ref={ref}>
          <div className="trace__drawing">
            <LineageDrawing lineage={lineage} exploded={inView && Boolean(lineage)} />
          </div>

          <div className="trace__sheet">
            <SpecBlock
              title="Model artifact"
              rows={[
                ["Artifact SHA-256", <Hash value={lineage?.model.artifact_sha256} />],
                ["Python", <span className="mono">{environment?.python_version ?? "—"}</span>],
                ["scikit-learn", <span className="mono">{environment?.scikit_learn_version ?? "—"}</span>],
                ["uv.lock SHA-256", <Hash value={environment?.lock_file_sha256} />],
              ]}
            />
            <SpecBlock
              title="Training run"
              rows={[
                ["MLflow run", <Hash value={lineage?.run.mlflow_run_id} head={8} tail={4} />],
                [
                  "Accuracy · F1",
                  <span className="mono">
                    {percent(lineage?.run.metrics?.accuracy)} · {percent(lineage?.run.metrics?.f1)}
                  </span>,
                ],
                [
                  "Git commit",
                  <span className="mono">
                    {lineage?.run.git_commit ? lineage.run.git_commit.slice(0, 10) : "—"}
                    {lineage?.run.git_dirty ? " (uncommitted changes)" : ""}
                  </span>,
                ],
                ["Features", <span className="mono">{lineage?.run.feature_columns?.length ?? "—"}</span>],
              ]}
            />
            <SpecBlock
              title="Configuration"
              rows={[
                ["Fingerprint", <Hash value={lineage?.config.fingerprint} />],
                ["Model", <span className="mono">{lineage?.config.model_type ?? "—"}</span>],
                [
                  "Parameters",
                  <span className="mono">
                    {parameters
                      ? Object.entries(parameters)
                          .map(([name, value]) => `${name} ${value ?? "none"}`)
                          .join(" · ")
                      : "—"}
                  </span>,
                ],
                [
                  "Split · seed",
                  <span className="mono">
                    {lineage?.config.test_size ?? "—"} · {lineage?.config.random_seed ?? "—"}
                  </span>,
                ],
              ]}
            />
            <SpecBlock
              title="Dataset"
              rows={[
                ["Fingerprint", <Hash value={lineage?.dataset.fingerprint} />],
                [
                  "Source",
                  <span className="mono">
                    {lineage?.dataset.original_filename ??
                      (lineage?.dataset.source === "iris" ? "Iris, built in" : lineage?.dataset.source ?? "—")}
                  </span>,
                ],
                [
                  "Shape · target",
                  <span className="mono">
                    {lineage?.dataset.row_count
                      ? `${lineage.dataset.row_count} × ${lineage.dataset.column_count} · ${
                          lineage.dataset.target_column ?? "—"
                        }`
                      : lineage?.dataset.source === "iris"
                        ? "built-in set"
                        : "—"}
                  </span>,
                ],
                ["File SHA-256", <Hash value={lineage?.dataset.file_sha256} />],
              ]}
            />
          </div>
        </div>
      )}
    </Panel>
  );
}
