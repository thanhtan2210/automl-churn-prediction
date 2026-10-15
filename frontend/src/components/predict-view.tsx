"use client";

import { useMutation } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { FileDrop } from "@/components/file-drop";
import { HowToRead } from "@/components/how-to-read";
import { ContactSlider } from "@/components/predict/contact-slider";
import { CustomerSheet } from "@/components/predict/customer-sheet";
import { CustomerTable, type Row } from "@/components/predict/customer-table";
import { KpiStrip } from "@/components/predict/kpi-strip";
import { RiskHistogram } from "@/components/predict/risk-histogram";
import { SegmentBreakdown } from "@/components/predict/segment-breakdown";
import { TopReasons } from "@/components/predict/top-reasons";
import { EmptyState, ErrorAlert, PageSkeleton } from "@/components/status";
import { Badge } from "@/components/ui/badge";
import { api, ApiError } from "@/lib/api";
import { pct } from "@/lib/format";
import { quartileEdges, ranks, reasonLabels, type Range } from "@/lib/predict-analysis";
import { useDataset } from "@/lib/queries";
import type { ModelMetrics, PredictResponse } from "@/lib/types";

function PredictFaq({ result }: { result: PredictResponse | undefined }) {
  return (
    <HowToRead
      intro="Scores come from the model trained on this dataset. Each customer is scored on the columns the model uses; other columns are ignored."
      items={[
        {
          q: "What does a 70% churn probability mean?",
          a: `It is the model's risk score. Use it to rank customers: higher means more likely to leave. The model was not checked for whether 70% really means 7 in 10 leave, so do not read it as an exact rate. Customers at or above ${
            result ? pct(result.threshold) : "the cut-off"
          } are flagged.`,
        },
        {
          q: "Why do so many customers share the same reason?",
          a: "A few columns, such as the contract type, move the risk score a lot for many customers. The reasons list the columns that push each customer's risk up the most, so the same strong column shows up again and again.",
        },
        {
          q: "Can I trust these numbers for a new month of data?",
          a: "Only if new customers look like the ones the model learned from. The estimates on this page come from customers kept aside during training. If prices, products or the customer mix change, retrain the model on recent data.",
        },
        {
          q: "Are the reasons the causes of churn?",
          a: "No. They show which values make the model rate this customer as riskier. They are a good starting point for a conversation, not proof of why someone leaves.",
        },
      ]}
    />
  );
}

export function PredictView({ id }: { id: string }) {
  const dataset = useDataset(id);
  const [fileName, setFileName] = useState<string | null>(null);
  const predict = useMutation({ mutationFn: (file: File) => api.predict(id, file) });

  if (dataset.isPending) return <PageSkeleton label="Loading the model…" />;
  if (dataset.isError) return <ErrorAlert error={dataset.error} title="Could not load the dataset" />;
  if (!dataset.data.model) {
    return (
      <EmptyState icon={Sparkles} title="No model yet">
        <Link href={`/datasets/${id}`} className="text-primary underline">
          Train a model first, then come back to score new customers.
        </Link>
      </EmptyState>
    );
  }

  const missing =
    predict.error instanceof ApiError && predict.error.status === 422
      ? ((predict.error.detail as { missing_columns?: string[] })?.missing_columns ?? [])
      : [];

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Score new customers</h1>
          <p className="text-muted-foreground">
            Upload customers with the {dataset.data.model.features.length} columns the model uses. Try{" "}
            <code className="text-xs">data/samples/*_new_customers.csv</code>: customers kept aside during training.
          </p>
        </div>
        <PredictFaq result={predict.data} />
      </div>

      <div className="space-y-3">
        <FileDrop
          compact={Boolean(predict.data)}
          onFile={(file) => {
            setFileName(file.name);
            predict.mutate(file);
          }}
          disabled={predict.isPending}
        />
        {predict.isPending && <PageSkeleton label={`Scoring ${fileName ?? "the file"}… large files take a few seconds.`} blocks={2} />}
        <ErrorAlert error={predict.error} title="This file could not be scored" />
        {missing.length > 0 && (
          <div className="flex flex-wrap gap-1 text-sm">
            <span className="text-muted-foreground">Missing columns:</span>
            {missing.map((m) => (
              <Badge key={m} variant="destructive">
                {m}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {predict.data && !predict.isPending && (
        <PredictResults
          key={`${fileName}-${predict.submittedAt}`}
          result={predict.data}
          metrics={dataset.data.model.metrics}
          fileName={fileName}
        />
      )}
    </div>
  );
}

function PredictResults({
  result,
  metrics,
  fileName,
}: {
  result: PredictResponse;
  metrics: ModelMetrics;
  fileName: string | null;
}) {
  const [topN, setTopN] = useState(result.n_flagged);
  const [selected, setSelected] = useState<number | null>(null);
  const [range, setRange] = useState<Range | null>(null);
  const [reason, setReason] = useState<string | null>(null);
  const rank = useMemo(() => ranks(result.predictions), [result]);
  const edges = useMemo(() => quartileEdges(result), [result]);
  const allRows: Row[] = useMemo(
    () => result.predictions.map((p, row) => ({ ...p, row, rank: rank[row] })),
    [result, rank],
  );
  const rows = useMemo(() => {
    let out = allRows;
    if (range) out = out.filter((p) => p.probability >= range[0] && (p.probability < range[1] || range[1] >= 1));
    if (reason) out = out.filter((p) => reasonLabels(p, result, edges).includes(reason));
    return out;
  }, [allRows, range, reason, result, edges]);

  return (
    <div className="space-y-8">
      <KpiStrip result={result} />
      <RiskHistogram result={result} selected={range} onSelect={setRange} />
      <TopReasons result={result} edges={edges} selected={reason} onSelect={setReason} />
      <ContactSlider result={result} metrics={metrics} topN={topN} onChange={setTopN} />
      <SegmentBreakdown result={result} metrics={metrics} />
      <CustomerTable
        result={result}
        rows={rows}
        allRows={allRows}
        topN={topN}
        range={range}
        reason={reason}
        onClearRange={() => setRange(null)}
        onClearReason={() => setReason(null)}
        onOpen={setSelected}
        fileName={fileName}
      />
      <CustomerSheet
        result={result}
        customer={selected === null ? null : result.predictions[selected]}
        rank={selected === null ? null : rank[selected]}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
