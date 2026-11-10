/**
 * Client for the ModelLore API.
 *
 * The console is served from the same origin as the API (FastAPI in
 * production, Vite's proxy in development), so every path is relative and no
 * CORS configuration is involved. The API key is passed in per call and never
 * stored here.
 */

export type ApiStatus = "checking" | "online" | "offline";

export type Page<T> = {
  items: T[];
  total: number;
  limit: number;
  offset: number;
};

export type Metrics = Record<string, number>;

export type Dataset = {
  dataset_id: string;
  name: string;
  original_filename: string;
  sha256: string;
  row_count: number | null;
  column_count: number | null;
  target_column: string | null;
  created_at: string | null;
};

export type Job = {
  job_id: string;
  dataset_id: string | null;
  config_name: string;
  status: string;
  mlflow_run_id: string | null;
  metrics: Metrics | null;
  model_id: string | null;
  lineage_run_id: string | null;
  error_message: string | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
};

export type Model = {
  model_id: string;
  training_job_id: string;
  name: string;
  version: string;
  metrics: Metrics | null;
  created_at: string | null;
  lineage_run_id: string | null;
  dataset_id: string | null;
  feature_columns: string[] | null;
  artifact_sha256: string | null;
};

export type Run = {
  mlflow_run_id: string;
  experiment_id: string;
  status: string;
  parameters: Record<string, string>;
  metrics: Metrics;
  tags: Record<string, string>;
};

export type UploadResult = {
  dataset_id: string;
  filename: string;
  size_bytes: number;
  sha256: string;
  column_count: number;
  row_count: number;
};

export type TrainRequest =
  | { config_name: "iris" }
  | { dataset_id: string; target_column?: string };

export type TrainResult = {
  status: string;
  lineage_run_id: string;
  mlflow_run_id: string;
  metrics: Metrics;
  parameters: Record<string, unknown>;
};

export type FeatureValue = number | string | boolean | null;

export type PredictResult = {
  lineage_run_id: string;
  predictions: Array<number | string>;
  count: number;
};

export type Lineage = {
  lineage_run_id: string | null;
  status: string | null;
  created_at: string | null;
  dataset: {
    fingerprint: string | null;
    source: string | null;
    target_column: string | null;
    dataset_id: string | null;
    original_filename: string | null;
    file_sha256: string | null;
    row_count: number | null;
    column_count: number | null;
  };
  config: {
    fingerprint: string | null;
    random_seed: number | null;
    test_size: number | null;
    model_type: string | null;
    task_type: string | null;
    parameters: Record<string, number | string | null> | null;
  };
  run: {
    mlflow_run_id: string | null;
    job_id: string | null;
    metrics: Metrics | null;
    feature_columns: string[] | null;
    git_commit: string | null;
    git_dirty: boolean | null;
  };
  model: {
    model_id: string | null;
    artifact_sha256: string | null;
    environment: {
      python_version: string | null;
      scikit_learn_version: string | null;
      lock_file_sha256: string | null;
    } | null;
  };
};

/** An error the API answered with, reduced to a status and a readable message. */
export class ApiError extends Error {
  readonly status: number;
  readonly expectedColumns: string[] | null;

  constructor(status: number, message: string, expectedColumns: string[] | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.expectedColumns = expectedColumns;
  }
}

/** Turn FastAPI's `detail` (a string, a validation list, or an object) into one message. */
export function describeDetail(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;

  if (typeof detail === "string") {
    return new ApiError(status, detail);
  }

  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: unknown; loc?: unknown } | undefined;
    const field = Array.isArray(first?.loc) ? first.loc.filter((part) => part !== "body").join(".") : "";
    const message = typeof first?.msg === "string" ? first.msg : "Request was not valid";
    return new ApiError(status, field ? `${field}: ${message}` : message);
  }

  if (detail && typeof detail === "object") {
    const { message, expected_columns: expected } = detail as {
      message?: unknown;
      expected_columns?: unknown;
    };
    const columns = Array.isArray(expected) ? expected.filter((c): c is string => typeof c === "string") : null;
    return new ApiError(status, typeof message === "string" ? message : "Request was not valid", columns);
  }

  const fallback: Record<number, string> = {
    401: "The API key was not accepted",
    429: "Too many requests; wait a moment and try again",
    503: "The API has no key configured",
  };
  return new ApiError(status, fallback[status] ?? `The API answered ${status}`);
}

type RequestOptions = {
  method?: "GET" | "POST";
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
};

async function request<T>(path: string, apiKey: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { "X-API-Key": apiKey };
  let body: BodyInit | undefined;

  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  } else if (options.form) {
    body = options.form;
  }

  let response: Response;
  try {
    response = await fetch(path, {
      method: options.method ?? "GET",
      headers,
      body,
      signal: options.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(0, "The API could not be reached");
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw describeDetail(response.status, payload);
  }
  return payload as T;
}

export async function fetchHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/health", { signal });
    return response.ok;
  } catch {
    return false;
  }
}

/** The API's own version, read from its public OpenAPI description. */
export async function fetchApiVersion(signal?: AbortSignal): Promise<string | null> {
  try {
    const response = await fetch("/openapi.json", { signal });
    if (!response.ok) {
      return null;
    }
    const spec: unknown = await response.json();
    const version = (spec as { info?: { version?: unknown } })?.info?.version;
    return typeof version === "string" ? version : null;
  } catch {
    return null;
  }
}

export const api = {
  /** A cheap authenticated call, used to check a key before keeping it. */
  verifyKey: (key: string) => request<Page<Dataset>>("/datasets?limit=1", key),

  listDatasets: (key: string) => request<Page<Dataset>>("/datasets?limit=100", key),
  listJobs: (key: string) => request<Page<Job>>("/jobs?limit=100", key),
  listModels: (key: string) => request<Page<Model>>("/models?limit=100", key),

  getRun: (key: string, mlflowRunId: string, signal?: AbortSignal) =>
    request<Run>(`/runs/${encodeURIComponent(mlflowRunId)}`, key, { signal }),

  getLineage: (key: string, lineageRunId: string, signal?: AbortSignal) =>
    request<Lineage>(`/lineage/${encodeURIComponent(lineageRunId)}`, key, { signal }),

  uploadDataset: (key: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResult>("/datasets/upload", key, { method: "POST", form });
  },

  train: (key: string, body: TrainRequest) =>
    request<TrainResult>("/train", key, { method: "POST", json: body }),

  predict: (key: string, lineageRunId: string, features: Array<Record<string, FeatureValue>>) =>
    request<PredictResult>("/predict", key, {
      method: "POST",
      json: { lineage_run_id: lineageRunId, features },
    }),
};
