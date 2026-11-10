import type { FeatureValue, Job, Model } from "./api";

/** sklearn's Iris targets, in the order the built-in dataset encodes them. */
const IRIS_CLASSES = ["setosa", "versicolor", "virginica"];

/** Jobs oldest first, so the waveform reads left to right like a tape. */
export function chronological(jobs: Job[]): Job[] {
  return [...jobs].sort((a, b) => {
    const left = a.created_at ?? "";
    const right = b.created_at ?? "";
    return left === right ? a.job_id.localeCompare(b.job_id) : left.localeCompare(right);
  });
}

/** Bar height for a run, between 0 and 1: its accuracy, or a short stub if it failed. */
export function barLevel(job: Job): number {
  if (job.status !== "FINISHED") {
    return 0.06;
  }
  const accuracy = job.metrics?.accuracy;
  if (typeof accuracy !== "number" || !Number.isFinite(accuracy)) {
    return 0.06;
  }
  return Math.min(1, Math.max(0.06, accuracy));
}

/** A run can be traced and played only if it left a model with lineage. */
export function isPlayable(job: Job | undefined, model: Model | undefined): boolean {
  return Boolean(job && job.status === "FINISHED" && model?.lineage_run_id && model.feature_columns?.length);
}

/** Read a prediction as a person would: Iris class indexes get their species name. */
export function describePrediction(model: Model | undefined, value: number | string): string {
  if (model?.name === "iris" && typeof value === "number" && IRIS_CLASSES[value]) {
    return `${value} · ${IRIS_CLASSES[value]}`;
  }
  return String(value);
}

/** Turn what was typed into a feature value: numbers become numbers, blanks become null. */
export function parseFeatureValue(raw: string): FeatureValue {
  const value = raw.trim();
  if (value === "") {
    return null;
  }
  if (/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(value)) {
    return Number(value);
  }
  if (value === "true" || value === "false") {
    return value === "true";
  }
  return value;
}

/** Starting values for the predict form, taken from Iris when the model is an Iris model. */
export function exampleFeatures(model: Model | undefined): Record<string, string> {
  const columns = model?.feature_columns ?? [];
  const iris: Record<string, string> = {
    "sepal length (cm)": "5.1",
    "sepal width (cm)": "3.5",
    "petal length (cm)": "1.4",
    "petal width (cm)": "0.2",
  };
  return Object.fromEntries(columns.map((column) => [column, iris[column] ?? ""]));
}
