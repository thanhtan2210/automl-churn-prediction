"use client";

import { useMutation } from "@tanstack/react-query";
import { Download } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { FileDrop } from "@/components/file-drop";
import { ErrorAlert, Loading } from "@/components/status";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, ApiError } from "@/lib/api";
import { num, pct } from "@/lib/format";
import { useDataset } from "@/lib/queries";
import type { PredictResponse, Reason, Strength } from "@/lib/types";
import { cn } from "@/lib/utils";

const STRENGTH_STYLE: Record<Strength, string> = {
  strong: "border-red-200 bg-red-50 text-red-800",
  moderate: "border-amber-200 bg-amber-50 text-amber-800",
  slight: "border-border bg-muted text-muted-foreground",
};

function ReasonChip({ reason }: { reason: Reason }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs",
        STRENGTH_STYLE[reason.strength],
      )}
    >
      <span>
        {reason.feature} = {reason.value}
      </span>
      <span className="font-medium">{reason.strength}</span>
    </span>
  );
}

function csvCell(value: string | number | boolean): string {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(result: PredictResponse, filename: string) {
  const header = ["id", "churn_probability", "flagged", "reason_1", "reason_2", "reason_3"];
  const lines = result.predictions.map((p) =>
    [
      p.id,
      p.probability.toFixed(4),
      p.flagged,
      ...[0, 1, 2].map((i) => {
        const r = p.reasons[i];
        return r ? `${r.feature} = ${r.value} (${r.strength})` : "";
      }),
    ]
      .map(csvCell)
      .join(","),
  );
  const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function PredictView({ id }: { id: string }) {
  const dataset = useDataset(id);
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const predict = useMutation({ mutationFn: (file: File) => api.predict(id, file) });

  const rows = useMemo(() => {
    const all = (predict.data?.predictions ?? [])
      .map((p, row) => ({ ...p, row }))
      .sort((a, b) => b.probability - a.probability);
    return flaggedOnly ? all.filter((p) => p.flagged) : all;
  }, [predict.data, flaggedOnly]);

  if (dataset.isPending) return <Loading label="Loading model…" />;
  if (dataset.isError) return <ErrorAlert error={dataset.error} title="Could not load the dataset" />;
  if (!dataset.data.model) {
    return (
      <Alert>
        <AlertTitle>No model yet</AlertTitle>
        <AlertDescription>
          <Link href={`/datasets/${id}`} className="underline">
            Train a model first.
          </Link>
        </AlertDescription>
      </Alert>
    );
  }

  const missing =
    predict.error instanceof ApiError && predict.error.status === 422
      ? ((predict.error.detail as { missing_columns?: string[] })?.missing_columns ?? [])
      : [];
  const result = predict.data;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Score new customers</CardTitle>
          <CardDescription>
            The file needs the {dataset.data.model.features.length} columns the model was trained on; other
            columns are ignored. Examples: <code className="text-xs">data/samples/*_new_customers.csv</code> (customers
            from the test split, never used for training).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDrop
            onFile={(file) => {
              setFileName(file.name);
              predict.mutate(file);
            }}
            disabled={predict.isPending}
          />
          {predict.isPending && <Loading label={`Scoring ${fileName ?? "file"}…`} />}
          <ErrorAlert error={predict.error} title="Could not score this file" />
          {missing.length > 0 && (
            <div className="flex flex-wrap gap-1 text-sm">
              <span className="text-muted-foreground">Missing:</span>
              {missing.map((m) => (
                <Badge key={m} variant="destructive">
                  {m}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {result && !predict.isPending && (
        <Card>
          <CardHeader>
            <CardTitle>
              {result.n_flagged.toLocaleString()} of {result.n_rows.toLocaleString()} customers flagged
            </CardTitle>
            <CardDescription>
              Flagged = churn probability at or above the model&apos;s threshold ({num(result.threshold, 2)}). Reasons
              are the columns that push this customer&apos;s risk up the most.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="size-4 accent-primary"
                  checked={flaggedOnly}
                  onChange={(e) => setFlaggedOnly(e.target.checked)}
                />
                Flagged only
              </label>
              <Button
                variant="outline"
                size="sm"
                onClick={() => downloadCsv(result, `churn_scores_${fileName?.replace(/\.[^.]+$/, "") ?? id}.csv`)}
              >
                <Download aria-hidden />
                Download CSV
              </Button>
              <span className="text-sm text-muted-foreground">{rows.length.toLocaleString()} rows shown</span>
            </div>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{result.id_column ?? "Row"}</TableHead>
                    <TableHead className="text-right">Churn probability</TableHead>
                    <TableHead>Risk</TableHead>
                    <TableHead>Top reasons</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((p) => (
                    <TableRow key={p.row}>
                      <TableCell className="font-medium">{p.id}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(p.probability, 1)}</TableCell>
                      <TableCell>
                        {p.flagged ? <Badge variant="destructive">flagged</Badge> : <Badge variant="secondary">low</Badge>}
                      </TableCell>
                      <TableCell className="whitespace-normal">
                        <div className="flex flex-wrap gap-1">
                          {p.reasons.length === 0 && <span className="text-xs text-muted-foreground">none</span>}
                          {p.reasons.map((r) => (
                            <ReasonChip key={r.feature} reason={r} />
                          ))}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
