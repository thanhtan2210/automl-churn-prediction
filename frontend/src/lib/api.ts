import type { Dataset, PredictResponse, TrainRequest, TrainResponse, UploadResponse } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: unknown,
  ) {
    super(message);
  }
}

/** Turns FastAPI's `detail` (a string, {message}, or a validation-error list) into one readable sentence. */
function detailMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === "object" && d && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
      .join("; ");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new ApiError(res.status, detailMessage(detail) ?? `Request failed (${res.status}).`, detail);
  }
  return res.json() as Promise<T>;
}

function fileForm(file: File): FormData {
  const form = new FormData();
  form.append("file", file);
  return form;
}

export const api = {
  uploadFile: (file: File) => request<UploadResponse>("/api/datasets", { method: "POST", body: fileForm(file) }),

  uploadSample: (sample: "telco" | "iranian") => {
    const form = new FormData();
    form.append("sample", sample);
    return request<UploadResponse>("/api/datasets", { method: "POST", body: form });
  },

  getDataset: (id: string, target?: string | null, positiveLabel?: string | null) => {
    const params = new URLSearchParams();
    if (target) params.set("target", target);
    if (target && positiveLabel) params.set("positive_label", positiveLabel);
    const query = params.size ? `?${params}` : "";
    return request<Dataset>(`/api/datasets/${encodeURIComponent(id)}${query}`);
  },

  train: (id: string, body: TrainRequest) =>
    request<TrainResponse>(`/api/datasets/${encodeURIComponent(id)}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  predict: (id: string, file: File) =>
    request<PredictResponse>(`/api/datasets/${encodeURIComponent(id)}/predict`, {
      method: "POST",
      body: fileForm(file),
    }),
};
