import { type KeyboardEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { ApiError, api, type Dataset, type Job, type Run } from "../../lib/api";
import { percent, serial, shortHash, tapeCounter } from "../../lib/format";
import { barLevel, chronological } from "../../lib/runs";
import { Panel } from "../Panel";

type PlaybackPanelProps = {
  apiKey: string;
  jobs: Job[];
  datasets: Dataset[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
};

const METRICS = ["accuracy", "precision", "recall", "f1"] as const;

function duration(job: Job | undefined): number {
  if (!job?.started_at || !job.completed_at) {
    return 0;
  }
  return Math.max(0, Date.parse(job.completed_at) - Date.parse(job.started_at));
}

function when(timestamp: string | null): string {
  if (!timestamp) {
    return "—";
  }
  const date = new Date(timestamp.endsWith("Z") || timestamp.includes("+") ? timestamp : `${timestamp}Z`);
  return Number.isNaN(date.getTime())
    ? timestamp
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Panel 03: every run as a bar; the orange playhead picks one and reads it out. */
export function PlaybackPanel({ apiKey, jobs, datasets, selectedJobId, onSelect }: PlaybackPanelProps) {
  const tape = useMemo(() => chronological(jobs), [jobs]);
  const index = tape.findIndex((job) => job.job_id === selectedJobId);
  const selected = index >= 0 ? tape[index] : undefined;
  const [run, setRun] = useState<Run | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const barRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const tapeRef = useRef<HTMLDivElement | null>(null);
  const [playheadX, setPlayheadX] = useState<number | null>(null);

  // The tape fills from the left, so the playhead follows the selected bar's real position.
  useLayoutEffect(() => {
    const tapeElement = tapeRef.current;
    const measure = () => {
      const slot = index >= 0 ? barRefs.current[index] : null;
      if (!tapeElement || !slot) {
        setPlayheadX(null);
        return;
      }
      const tapeBox = tapeElement.getBoundingClientRect();
      const slotBox = slot.getBoundingClientRect();
      setPlayheadX(slotBox.left - tapeBox.left + slotBox.width / 2);
    };
    measure();
    if (!tapeElement || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(tapeElement);
    return () => observer.disconnect();
  }, [index, tape.length]);

  useEffect(() => {
    setRun(null);
    setRunError(null);
    if (!selected?.mlflow_run_id) {
      return;
    }
    const controller = new AbortController();
    api
      .getRun(apiKey, selected.mlflow_run_id, controller.signal)
      .then(setRun)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setRunError(error instanceof ApiError ? error.message : "The MLflow run could not be read");
        }
      });
    return () => controller.abort();
  }, [apiKey, selected?.mlflow_run_id]);

  const move = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity }[event.key];
    if (step === undefined || tape.length === 0) {
      return;
    }
    event.preventDefault();
    const next = Math.min(tape.length - 1, Math.max(0, (index < 0 ? tape.length - 1 : index) + step));
    onSelect(tape[next].job_id);
    barRefs.current[next]?.focus();
  };

  if (tape.length === 0) {
    return (
      <Panel index="03" name="Playback" state="Blank tape" className="panel--wide">
        <p className="panel__detail">No runs yet. Record one above and it appears here as the first bar.</p>
      </Panel>
    );
  }

  const dataset = datasets.find((item) => item.dataset_id === selected?.dataset_id);

  return (
    <Panel
      index="03"
      name="Playback"
      state={`${serial(tape.length, 3)} runs`}
      className="panel--wide"
      id="playback"
    >
      <div className="tape" ref={tapeRef} onKeyDown={move}>
        <div className="tape__bars" role="listbox" aria-label="Training runs, oldest first" aria-orientation="horizontal">
          {tape.map((job, position) => {
            const isSelected = job.job_id === selectedJobId;
            return (
              <button
                key={job.job_id}
                ref={(element) => {
                  barRefs.current[position] = element;
                }}
                type="button"
                role="option"
                aria-selected={isSelected}
                tabIndex={isSelected || (index < 0 && position === tape.length - 1) ? 0 : -1}
                className="tape__slot"
                data-status={job.status}
                aria-label={`Run ${position + 1}, ${job.config_name}, ${job.status.toLowerCase()}${
                  job.metrics?.accuracy !== undefined ? `, accuracy ${percent(job.metrics.accuracy)}` : ""
                }`}
                onClick={() => onSelect(job.job_id)}
              >
                <span className="tape__bar" style={{ height: `${barLevel(job) * 100}%` }} />
              </button>
            );
          })}
        </div>

        {playheadX !== null && (
          <div className="tape__playhead" style={{ left: `${playheadX}px` }} aria-hidden="true">
            <svg viewBox="0 0 24 12" className="tape__cap tape__cap--top">
              <path d="M0 0 C8 0 10 4 12 12 C14 4 16 0 24 0 Z" fill="var(--signal)" />
            </svg>
            <svg viewBox="0 0 24 12" className="tape__cap tape__cap--bottom">
              <path d="M0 12 C8 12 10 8 12 0 C14 8 16 12 24 12 Z" fill="var(--signal)" />
            </svg>
          </div>
        )}

        {selected && (
          <div className="chip mono" aria-hidden="true">
            <span>{tapeCounter(duration(selected))}</span>
            <span className="chip__tag">{selected.status === "FINISHED" ? "REC" : "ERR"}</span>
            <span>
              run {serial(index + 1, 3)}/{serial(tape.length, 3)}
            </span>
            <span>{selected.config_name}</span>
          </div>
        )}
      </div>

      {selected ? (
        <div className="playback">
          <dl className="meters">
            {METRICS.map((name) => (
              <div key={name} className="meter">
                <dt className="label">{name}</dt>
                <dd className="display meter__value">{percent(selected.metrics?.[name])}</dd>
              </div>
            ))}
          </dl>

          <dl className="specs">
            <div>
              <dt className="label">Status</dt>
              <dd className="mono">{selected.status}</dd>
            </div>
            <div>
              <dt className="label">Data</dt>
              <dd className="mono">{dataset ? dataset.original_filename : selected.dataset_id ? "Uploaded CSV" : "Iris, built in"}</dd>
            </div>
            <div>
              <dt className="label">Recorded</dt>
              <dd className="mono">{when(selected.created_at)}</dd>
            </div>
            <div>
              <dt className="label">MLflow run</dt>
              <dd className="mono" title={selected.mlflow_run_id ?? undefined}>
                {shortHash(selected.mlflow_run_id, 8, 4)} {run ? `· ${run.status.toLowerCase()}` : ""}
              </dd>
            </div>
            {run && Object.keys(run.parameters).length > 0 && (
              <div className="specs__wide">
                <dt className="label">Parameters</dt>
                <dd className="mono">
                  {Object.entries(run.parameters)
                    .filter(([name]) => !["dataset_name", "task_type"].includes(name))
                    .map(([name, value]) => `${name} ${value}`)
                    .join(" · ")}
                </dd>
              </div>
            )}
            {selected.error_message && (
              <div className="specs__wide">
                <dt className="label">Error</dt>
                <dd>{selected.error_message}</dd>
              </div>
            )}
          </dl>
          {runError && <p className="notice">{runError}</p>}
        </div>
      ) : (
        <p className="panel__detail">Pick a bar to read out that run.</p>
      )}
    </Panel>
  );
}
