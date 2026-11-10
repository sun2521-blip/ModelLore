import { describe, expect, it } from "vitest";

import type { Job, Model } from "./api";
import { barLevel, chronological, describePrediction, exampleFeatures, isPlayable, parseFeatureValue } from "./runs";

function job(overrides: Partial<Job> = {}): Job {
  return {
    job_id: "job",
    dataset_id: null,
    config_name: "iris",
    status: "FINISHED",
    mlflow_run_id: null,
    metrics: { accuracy: 0.9 },
    model_id: "model",
    lineage_run_id: "lineage",
    error_message: null,
    created_at: "2026-11-10T10:00:00",
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

function model(overrides: Partial<Model> = {}): Model {
  return {
    model_id: "model",
    training_job_id: "job",
    name: "iris",
    version: "1",
    metrics: null,
    created_at: null,
    lineage_run_id: "lineage",
    dataset_id: null,
    feature_columns: ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    artifact_sha256: "a".repeat(64),
    ...overrides,
  };
}

describe("chronological", () => {
  it("orders runs oldest first without changing the input", () => {
    const jobs = [job({ job_id: "b", created_at: "2026-11-10T12:00:00" }), job({ job_id: "a", created_at: "2026-11-10T09:00:00" })];
    expect(chronological(jobs).map((item) => item.job_id)).toEqual(["a", "b"]);
    expect(jobs[0].job_id).toBe("b");
  });
});

describe("barLevel", () => {
  it("uses accuracy for finished runs and a stub for the rest", () => {
    expect(barLevel(job({ metrics: { accuracy: 0.9 } }))).toBe(0.9);
    expect(barLevel(job({ status: "FAILED", metrics: null }))).toBe(0.06);
    expect(barLevel(job({ metrics: {} }))).toBe(0.06);
    expect(barLevel(job({ metrics: { accuracy: 0 } }))).toBe(0.06);
  });
});

describe("isPlayable", () => {
  it("needs a finished job and a model with lineage and features", () => {
    expect(isPlayable(job(), model())).toBe(true);
    expect(isPlayable(job({ status: "FAILED" }), model())).toBe(false);
    expect(isPlayable(job(), model({ lineage_run_id: null }))).toBe(false);
    expect(isPlayable(job(), model({ feature_columns: [] }))).toBe(false);
    expect(isPlayable(undefined, undefined)).toBe(false);
  });
});

describe("describePrediction", () => {
  it("names Iris classes and leaves other labels alone", () => {
    expect(describePrediction(model(), 0)).toBe("0 · setosa");
    expect(describePrediction(model(), 2)).toBe("2 · virginica");
    expect(describePrediction(model({ name: "csv-test" }), "setosa")).toBe("setosa");
    expect(describePrediction(model(), 7)).toBe("7");
  });
});

describe("parseFeatureValue", () => {
  it("turns typed text into the right JSON value", () => {
    expect(parseFeatureValue("5.1")).toBe(5.1);
    expect(parseFeatureValue("-2")).toBe(-2);
    expect(parseFeatureValue("1e3")).toBe(1000);
    expect(parseFeatureValue("  ")).toBeNull();
    expect(parseFeatureValue("true")).toBe(true);
    expect(parseFeatureValue("red")).toBe("red");
    expect(parseFeatureValue("5.1.2")).toBe("5.1.2");
  });
});

describe("exampleFeatures", () => {
  it("prefills Iris values and leaves unknown columns blank", () => {
    expect(exampleFeatures(model())["petal width (cm)"]).toBe("0.2");
    expect(exampleFeatures(model({ feature_columns: ["colour"] }))).toEqual({ colour: "" });
  });
});
