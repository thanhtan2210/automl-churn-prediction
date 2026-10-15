"use client";

import { useQuery } from "@tanstack/react-query";

import { api } from "./api";

export const datasetKey = (id: string) => ["dataset", id] as const;

/** Dataset status; polls every 2 s while a model is training. */
export function useDataset(id: string) {
  return useQuery({
    queryKey: datasetKey(id),
    queryFn: () => api.getDataset(id),
    refetchInterval: (query) => (query.state.data?.status === "training" ? 2000 : false),
  });
}

/** Profile for one (target, positive label) pair; computed by the API on the training split and cached there. */
export function useProfile(id: string, target: string | null, positiveLabel: string | null) {
  return useQuery({
    queryKey: ["profile", id, target, positiveLabel] as const,
    queryFn: () => api.getDataset(id, target, positiveLabel),
    enabled: Boolean(target && positiveLabel),
    staleTime: Infinity,
  });
}
