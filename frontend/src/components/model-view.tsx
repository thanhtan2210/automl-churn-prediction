"use client";

import { ArrowRight, BadgeCheck, Crosshair, Loader2, Scale, Sparkles, Target } from "lucide-react";
import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { HowToRead } from "@/components/how-to-read";
import { Kpi } from "@/components/kpi";
import { EmptyState, ErrorAlert, PageSkeleton } from "@/components/status";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { buttonVariants } from "@/components/ui/button";
import { ci, duration, num, pct, signed } from "@/lib/format";
import { useDataset } from "@/lib/queries";
import type { Dataset, Interval, ModelMetrics } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODEL = "var(--chart-model)";
const BASELINE = "var(--chart-baseline)";
// A difference whose CI is above zero but smaller than this is called "slightly better".
const SMALL_EDGE = 0.02;

export function verdict(diff: number, [lo, hi]: Interval): string {
  if (lo > 0 && diff < SMALL_EDGE) return "LightGBM is slightly better than the baseline";
  if (lo > 0) return "LightGBM is better than the baseline";
  if (hi < 0) return "The baseline is better than LightGBM";
  return "No clear difference between LightGBM and the baseline";
}

export function ModelView({ id }: { id: string }) {
  const { data, isPending, isError, error } = useDataset(id);
  if (isPending) return <PageSkeleton label="Loading the model…" />;
  if (isError) return <ErrorAlert error={error} title="Could not load the dataset" />;
  if (data.status === "training") return <TrainingProgress dataset={data} />;
  if (data.status === "failed") {
    return (
      <div className="space-y-3">
        <ErrorAlert error={data.error ?? "Training failed."} title="Training failed" />
        <Link href={`/datasets/${id}`} className={buttonVariants({ variant: "outline" })}>
          Back to the columns
        </Link>
      </div>
    );
  }
  if (!data.model) {
    return (
      <EmptyState icon={Sparkles} title="No model yet">
        <Link href={`/datasets/${id}`} className="text-primary underline">
          Choose the column that says who left and the columns to keep, then train.
        </Link>
      </EmptyState>
    );
  }
  return <ModelReport id={id} metrics={data.model.metrics} />;
}

const TRAINING_STEPS = [
  ["Try settings", "Each setting is scored on the training customers only, by cross-checking parts of them against each other."],
  ["Fit the final model", "The best setting is trained on all training customers, next to a simple baseline model."],
  ["Test once", "Both models score the customers kept aside, which they have never seen."],
];

function TrainingProgress({ dataset }: { dataset: Dataset }) {
  const p = dataset.training_progress ?? null;
  const share = p && p.trials_total ? p.trials_done / p.trials_total : null;
  const finishing = p !== null && p.trials_done >= p.trials_total;
  return (
    <div className="space-y-4 rounded-2xl border bg-card p-6 shadow-xs">
      <div className="flex items-center gap-3">
        <Loader2 className="size-6 animate-spin text-primary" aria-hidden />
        <div>
          <p className="text-lg font-semibold">Training the model…</p>
          <p className="text-sm text-muted-foreground">
            {finishing
              ? "Settings chosen. Fitting the final model and testing it on customers it has not seen."
              : "Trying different model settings and keeping the best one."}{" "}
            Running for {duration(dataset.training_elapsed_seconds ?? 0)}.
          </p>
        </div>
      </div>
      <div
        className="h-3 overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={p?.trials_total ?? 0}
        aria-valuenow={p?.trials_done ?? 0}
      >
        <div
          className={cn("h-full rounded-full bg-primary transition-all", share === null && "w-1/3 animate-pulse")}
          style={share === null ? undefined : { width: `${Math.max(3, share * 100)}%` }}
        />
      </div>
      {p && (
        <p className="text-sm tabular-nums text-muted-foreground">
          {p.trials_done} of {p.trials_total} settings tried. This page updates every 2 seconds.
        </p>
      )}
      <ol className="grid gap-3 pt-2 text-sm sm:grid-cols-3">
        {TRAINING_STEPS.map(([title, text], i) => (
          <li key={title} className="rounded-xl bg-muted/60 p-4">
            <p className="font-medium">
              {i + 1}. {title}
            </p>
            <p className="mt-1 text-muted-foreground">{text}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}

function ModelFaq({ metrics }: { metrics: ModelMetrics }) {
  const { test } = metrics;
  const at = test.at_threshold_lightgbm;
  const auc = test.lightgbm.roc_auc;
  const v = verdict(test.difference_lightgbm_minus_lr.roc_auc, test.difference_lightgbm_minus_lr.roc_auc_ci95);
  return (
    <HowToRead
      intro={`All numbers here come from ${test.n.toLocaleString()} customers the model never saw while learning.`}
      items={[
        {
          q: `Is ${pct(auc)} good?`,
          a: `If you pick one customer who left and one who stayed, the model gives the leaver the higher risk ${pct(auc)} of the time. Guessing would get 50%, a perfect model 100%. The simple baseline gets ${pct(test.logistic_regression.roc_auc)}: ${v.toLowerCase()}.`,
        },
        {
          q: "What is the baseline and why compare with it?",
          a: "A simple model that adds up a fixed effect for each column. If the more complex model cannot clearly beat it, the simple one is just as useful and easier to explain.",
        },
        {
          q: "Why does the model flag so many customers?",
          a: `It flags ${at.n_flagged.toLocaleString()} of ${test.n.toLocaleString()} test customers (${pct(at.n_flagged / test.n)}). The cut-off was chosen on the training data to balance catching leavers against false alarms. If your team can contact fewer people, use the slider on the Predict page.`,
        },
        {
          q: "What drives churn — does a long bar mean 'causes'?",
          a: "No. A long bar means the column moves the model's risk score a lot, up or down. It shows what the model relies on, not what causes customers to leave.",
        },
        {
          q: "Can I trust the test numbers?",
          a: "They are measured on customers kept aside from training. If you retrain many times on the same file with different columns, the same test customers are reused, so the numbers become a little too optimistic.",
        },
      ]}
    />
  );
}

function ModelReport({ id, metrics }: { id: string; metrics: ModelMetrics }) {
  const { test, training, curves } = metrics;
  const diff = test.difference_lightgbm_minus_lr;
  const at = test.at_threshold_lightgbm;
  const nChurners = Math.round(test.n * test.positive_rate);
  const importance = metrics.global_importance_top10;
  const outOfTen = Math.round(at.precision * 10);

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <span className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
            <BadgeCheck className="size-6" aria-hidden />
          </span>
          <div>
            <p className="text-2xl font-semibold tracking-tight sm:text-3xl">{verdict(diff.roc_auc, diff.roc_auc_ci95)}.</p>
            <p className="text-muted-foreground">
              Tested on {test.n.toLocaleString()} customers the model never saw while learning.
            </p>
          </div>
        </div>
        <ModelFaq metrics={metrics} />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Kpi
          icon={Target}
          value={pct(at.recall)}
          label="of customers who left are caught"
          hint={`${Math.round(at.recall * nChurners).toLocaleString()} of the ${nChurners.toLocaleString()} test customers who left were flagged.`}
        />
        <Kpi
          icon={Crosshair}
          value={`${outOfTen} of 10`}
          label="flagged customers really left"
          hint={`${pct(at.precision)} of the ${at.n_flagged.toLocaleString()} flagged test customers left.`}
        />
        <Kpi
          icon={Scale}
          value={pct(test.lightgbm.roc_auc)}
          label="of the time a leaver is ranked above a stayer"
          hint={`Simple baseline: ${pct(test.logistic_regression.roc_auc)}. Guessing: 50%.`}
        />
      </div>

      <section className="space-y-3 rounded-2xl border bg-card p-6 shadow-xs">
        <div>
          <h2 className="text-lg font-semibold">What drives churn in this data</h2>
          <p className="text-sm text-muted-foreground">
            How much each column moves a customer&apos;s risk score on average, up or down. It shows what the model
            relies on, not what causes customers to leave.
          </p>
        </div>
        <BarChart
          responsive
          style={{ width: "100%", height: Math.max(260, importance.length * 36) }}
          data={importance}
          layout="vertical"
          margin={{ top: 0, right: 48, bottom: 0, left: 10 }}
        >
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 13 }} tickLine={false} axisLine={false} />
          <Tooltip formatter={(value) => num(Number(value), 2)} cursor={{ fill: "var(--muted)" }} />
          <Bar dataKey="mean_abs_contribution" name="Average effect" fill={MODEL} radius={[0, 6, 6, 0]} barSize={20} isAnimationActive={false}>
            <LabelList dataKey="mean_abs_contribution" position="right" formatter={(v) => num(Number(v), 2)} className="fill-muted-foreground text-xs" />
          </Bar>
        </BarChart>
      </section>

      <Accordion className="rounded-2xl border bg-card px-6 shadow-xs">
        <AccordionItem value="tech">
          <AccordionTrigger className="text-base">Technical details</AccordionTrigger>
          <AccordionContent className="space-y-6">
            <div className="grid gap-4 md:grid-cols-3">
              <ScoreBlock title="ROC-AUC" lgbm={test.lightgbm.roc_auc} lr={test.logistic_regression.roc_auc} lgbmCi={test.lightgbm.roc_auc_ci95} lrCi={test.logistic_regression.roc_auc_ci95} />
              <ScoreBlock title="PR-AUC" lgbm={test.lightgbm.pr_auc} lr={test.logistic_regression.pr_auc} lgbmCi={test.lightgbm.pr_auc_ci95} lrCi={test.logistic_regression.pr_auc_ci95} />
              <div className="rounded-xl border p-4 text-sm">
                <p className="text-muted-foreground">LightGBM − LR (paired bootstrap)</p>
                <p className="tabular-nums">
                  ROC-AUC {signed(diff.roc_auc)} <span className="text-muted-foreground">95% CI {ci(diff.roc_auc_ci95)}</span>
                </p>
                <p className="tabular-nums">
                  PR-AUC {signed(diff.pr_auc)} <span className="text-muted-foreground">95% CI {ci(diff.pr_auc_ci95)}</span>
                </p>
              </div>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <CurveChart title="ROC curve" xKey="fpr" yKey="tpr" xLabel="False-positive rate"
                lgbm={curves.lightgbm.roc as Point[]} lr={curves.logistic_regression.roc as Point[]} diagonal />
              <CurveChart title="Precision–recall curve" xKey="recall" yKey="precision" xLabel="Recall"
                lgbm={curves.lightgbm.pr as Point[]} lr={curves.logistic_regression.pr as Point[]} baseline={test.positive_rate} />
            </div>
            <ul className="space-y-1 text-sm text-muted-foreground">
              <li>
                Threshold {num(at.threshold, 2)} (best F1 on out-of-fold training predictions): flags{" "}
                {at.n_flagged.toLocaleString()} of {test.n.toLocaleString()} test customers, precision {pct(at.precision)},
                recall {pct(at.recall)}.
              </li>
              <li>
                Optuna: {training.optuna.n_trials_completed} of {training.optuna.n_trials_requested} trials
                {training.optuna.stopped_by_timeout ? " (stopped by the time limit)" : ""}; 5-fold CV ROC-AUC{" "}
                {num(training.lightgbm.cv_auc_mean)} ± {num(training.lightgbm.cv_auc_std)} on {metrics.n_train.toLocaleString()}{" "}
                training rows (baseline {num(training.logistic_regression.cv_auc_mean)}).
              </li>
              {metrics.notes.map((n) => (
                <li key={n}>Note: {n}</li>
              ))}
            </ul>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Link href={`/datasets/${id}/predict`} className={cn(buttonVariants({ size: "lg" }))}>
        Score new customers
        <ArrowRight aria-hidden />
      </Link>
    </div>
  );
}

function ScoreBlock({ title, lgbm, lr, lgbmCi, lrCi }: { title: string; lgbm: number; lr: number; lgbmCi: Interval; lrCi: Interval }) {
  return (
    <div className="rounded-xl border p-4 text-sm">
      <p className="text-muted-foreground">{title} on the test split</p>
      <p className="tabular-nums">
        <span className="font-medium" style={{ color: MODEL }}>LightGBM</span> {num(lgbm)}{" "}
        <span className="text-muted-foreground">95% CI {ci(lgbmCi)}</span>
      </p>
      <p className="tabular-nums">
        <span className="font-medium">Baseline (LR)</span> {num(lr)} <span className="text-muted-foreground">95% CI {ci(lrCi)}</span>
      </p>
    </div>
  );
}

type Point = Record<string, number>;

function CurveChart({
  title,
  xKey,
  yKey,
  xLabel,
  lgbm,
  lr,
  diagonal,
  baseline,
}: {
  title: string;
  xKey: string;
  yKey: string;
  xLabel: string;
  lgbm: Point[];
  lr: Point[];
  diagonal?: boolean;
  baseline?: number;
}) {
  return (
    <div className="rounded-xl border p-4">
      <p className="mb-2 text-sm font-medium">{title}</p>
      <LineChart responsive style={{ width: "100%", height: 260 }} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey={xKey} domain={[0, 1]} tickCount={6} label={{ value: xLabel, position: "insideBottom", offset: -12 }} />
        <YAxis type="number" domain={[0, 1]} tickCount={6} />
        <Tooltip formatter={(value) => num(Number(value), 3)} labelFormatter={(l) => `${xLabel} ${num(Number(l), 3)}`} />
        <Legend verticalAlign="top" height={28} />
        {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#bbb" strokeDasharray="4 4" />}
        {baseline !== undefined && <ReferenceLine y={baseline} stroke="#bbb" strokeDasharray="4 4" />}
        <Line data={lgbm} dataKey={yKey} name="LightGBM" stroke={MODEL} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line data={lr} dataKey={yKey} name="Baseline (LR)" stroke={BASELINE} dot={false} strokeWidth={2} isAnimationActive={false} />
      </LineChart>
    </div>
  );
}
