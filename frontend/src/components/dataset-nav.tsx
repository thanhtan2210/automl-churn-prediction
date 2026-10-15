"use client";

import { FileSpreadsheet } from "lucide-react";

import { StepBar } from "@/components/step-bar";
import { Badge } from "@/components/ui/badge";
import { useDataset } from "@/lib/queries";

const STATUS_LABEL = {
  uploaded: "uploaded",
  profiled: "columns checked",
  training: "training…",
  trained: "model ready",
  failed: "training failed",
} as const;

/** File name, size and status, then the step bar, on every dataset page. */
export function DatasetNav({ id }: { id: string }) {
  const { data } = useDataset(id);
  return (
    <>
      <div className="mb-4 flex min-w-0 flex-wrap items-center gap-2">
        <FileSpreadsheet className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <span className="truncate font-medium">{data?.filename ?? "Loading dataset…"}</span>
        {data && (
          <>
            <span className="text-sm text-muted-foreground">{data.n_rows.toLocaleString()} rows</span>
            <Badge variant={data.status === "failed" ? "destructive" : "secondary"}>{STATUS_LABEL[data.status]}</Badge>
          </>
        )}
      </div>
      <StepBar dataset={data} id={id} />
    </>
  );
}
