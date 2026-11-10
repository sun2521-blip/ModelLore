import { useCallback, useEffect, useState } from "react";

import { ApiError, api, type Dataset, type Job, type Model } from "../lib/api";

type ConsoleData = {
  datasets: Dataset[];
  jobs: Job[];
  models: Model[];
  loading: boolean;
  error: ApiError | null;
};

const EMPTY: ConsoleData = { datasets: [], jobs: [], models: [], loading: false, error: null };

/** Datasets, training jobs, and models for the console, reloaded on demand. */
export function useConsoleData(apiKey: string | null) {
  const [data, setData] = useState<ConsoleData>(EMPTY);

  const reload = useCallback(async () => {
    if (!apiKey) {
      setData(EMPTY);
      return;
    }

    setData((current) => ({ ...current, loading: true, error: null }));
    try {
      const [datasets, jobs, models] = await Promise.all([
        api.listDatasets(apiKey),
        api.listJobs(apiKey),
        api.listModels(apiKey),
      ]);
      setData({
        datasets: datasets.items,
        jobs: jobs.items,
        models: models.items,
        loading: false,
        error: null,
      });
    } catch (error) {
      setData((current) => ({
        ...current,
        loading: false,
        error: error instanceof ApiError ? error : new ApiError(0, "The console could not load its data"),
      }));
    }
  }, [apiKey]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { ...data, reload };
}

export type ConsoleDataState = ReturnType<typeof useConsoleData>;
