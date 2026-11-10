import { useCallback, useEffect, useMemo, useState } from "react";

import type { ApiStatus, TrainResult, UploadResult } from "../../lib/api";
import type { ConsoleDataState } from "../../hooks/useConsoleData";
import { chronological } from "../../lib/runs";
import { AccessPanel } from "./AccessPanel";
import { PlayPanel } from "./PlayPanel";
import { PlaybackPanel } from "./PlaybackPanel";
import { RecordPanel } from "./RecordPanel";
import { TracePanel } from "./TracePanel";
import { UploadPanel } from "./UploadPanel";

type ConsoleProps = {
  status: ApiStatus;
  apiKey: string | null;
  onKey: (apiKey: string | null, keepForTab?: boolean) => void;
  data: ConsoleDataState;
};

/** The working console: access, insert, record, playback, trace, play. */
export function Console({ status, apiKey, onKey, data }: ConsoleProps) {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [knownColumns, setKnownColumns] = useState<Record<string, string[]>>({});
  const [freshDatasetId, setFreshDatasetId] = useState<string | null>(null);
  const [pendingLineageId, setPendingLineageId] = useState<string | null>(null);
  const { reload } = data;

  // A rejected key (rotated on the server, say) sends the console back to Access.
  useEffect(() => {
    if (data.error?.status === 401) {
      onKey(null);
    }
  }, [data.error, onKey]);

  // Select the newest run by default, or the one just recorded once it shows up.
  useEffect(() => {
    if (data.jobs.length === 0) {
      return;
    }
    if (pendingLineageId) {
      const recorded = data.jobs.find((job) => job.lineage_run_id === pendingLineageId);
      if (recorded) {
        setSelectedJobId(recorded.job_id);
        setPendingLineageId(null);
        return;
      }
    }
    if (!selectedJobId || !data.jobs.some((job) => job.job_id === selectedJobId)) {
      const tape = chronological(data.jobs);
      setSelectedJobId(tape[tape.length - 1].job_id);
    }
  }, [data.jobs, pendingLineageId, selectedJobId]);

  const selectedJob = useMemo(() => data.jobs.find((job) => job.job_id === selectedJobId), [data.jobs, selectedJobId]);
  const selectedModel = useMemo(
    () => data.models.find((model) => model.model_id === selectedJob?.model_id),
    [data.models, selectedJob],
  );

  const onUploaded = useCallback(
    (result: UploadResult, columns: string[]) => {
      setKnownColumns((current) => ({ ...current, [result.dataset_id]: columns }));
      setFreshDatasetId(result.dataset_id);
      void reload();
    },
    [reload],
  );

  const onRecorded = useCallback(
    (result: TrainResult) => {
      setPendingLineageId(result.lineage_run_id);
      void reload();
    },
    [reload],
  );

  if (status === "offline") {
    return (
      <p className="notice notice--error console__offline" role="status">
        The API is not answering. Start it with <span className="mono">uv run uvicorn api.main:app</span>, then this
        page reconnects on its own.
      </p>
    );
  }

  if (!apiKey) {
    return (
      <div className="module-grid">
        <AccessPanel onConnect={(key, keep) => onKey(key, keep)} />
      </div>
    );
  }

  return (
    <>
      <div className="console__bar">
        <p className="label">
          <span className="rec-dot" data-live="true" aria-hidden="true" /> Connected
          {data.loading ? " · reading" : ""}
        </p>
        <button type="button" className="button button--quiet label" onClick={() => onKey(null)}>
          Forget key
        </button>
      </div>

      {data.error && data.error.status !== 401 && (
        <p className="notice notice--error" role="alert">
          {data.error.message}
        </p>
      )}

      <div className="module-grid">
        <UploadPanel apiKey={apiKey} onUploaded={onUploaded} />
        <RecordPanel
          apiKey={apiKey}
          datasets={data.datasets}
          knownColumns={knownColumns}
          preferredDatasetId={freshDatasetId}
          onRecorded={onRecorded}
        />
        <PlaybackPanel
          apiKey={apiKey}
          jobs={data.jobs}
          datasets={data.datasets}
          selectedJobId={selectedJobId}
          onSelect={setSelectedJobId}
        />
        <TracePanel apiKey={apiKey} job={selectedJob} />
        <PlayPanel apiKey={apiKey} job={selectedJob} model={selectedModel} />
      </div>
    </>
  );
}
