"use client";

import { ArrowRight, Loader2 } from "lucide-react";
import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ErrorAlert, Loading } from "@/components/status";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ci, duration, num, pct, signed } from "@/lib/format";
import { useDataset } from "@/lib/queries";
import type { Interval, ModelMetrics } from "@/lib/types";

const MODEL = "var(--chart-model)";
const BASELINE = "var(--chart-baseline)";

export function ModelView({ id }: { id: string }) {
  const { data, isPending, isError, error } = useDataset(id);
  if (isPending) return <Loading label="Loading model…" />;
  if (isError) return <ErrorAlert error={error} title="Could not load the dataset" />;

  if (data.status === "training") {
    return (
      <Card>
        <CardContent className="flex items-center gap-3 py-8">
          <Loader2 className="size-5 animate-spin text-muted-foreground" aria-hidden />
          <div>
            <p className="font-medium">Training LightGBM with Optuna and the logistic-regression baseline…</p>
            <p className="text-sm text-muted-foreground">
              Running for {duration(data.training_elapsed_seconds ?? 0)}. This page updates every 2 seconds.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (data.status === "failed") {
    return (
      <div className="space-y-3">
        <ErrorAlert error={data.error ?? "Training failed."} title="Training failed" />
        <Link href={`/datasets/${id}`} className={buttonVariants({ variant: "outline" })}>
          Back to the profile
        </Link>
      </div>
    );
  }
  if (!data.model) {
    return (
      <Alert>
        <AlertTitle>No model yet</AlertTitle>
        <AlertDescription>
          <Link href={`/datasets/${id}`} className="underline">
            Choose the churn column and the columns to keep, then train.
          </Link>
        </AlertDescription>
      </Alert>
    );
  }
  return <ModelReport id={id} metrics={data.model.metrics} />;
}

function verdict([lo, hi]: Interval): { text: string; tone: string } {
  if (lo > 0) return { text: "LightGBM is better than the baseline", tone: "text-emerald-700" };
  if (hi < 0) return { text: "The baseline is better than LightGBM", tone: "text-amber-700" };
  return { text: "No clear difference between LightGBM and the baseline", tone: "text-foreground" };
}

function ScoreCard({ title, lgbm, lr, lgbmCi, lrCi, note }: {
  title: string;
  lgbm: number;
  lr: number;
  lgbmCi: Interval;
  lrCi: Interval;
  note: string;
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardDescription>{title} on the test split</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{num(lgbm)}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 text-sm">
        <p className="tabular-nums">
          <span className="font-medium" style={{ color: MODEL }}>LightGBM</span> {num(lgbm)}{" "}
          <span className="text-muted-foreground">95% CI {ci(lgbmCi)}</span>
        </p>
        <p className="tabular-nums">
          <span className="font-medium" style={{ color: BASELINE }}>Baseline (LR)</span> {num(lr)}{" "}
          <span className="text-muted-foreground">95% CI {ci(lrCi)}</span>
        </p>
        <p className="text-xs text-muted-foreground">{note}</p>
      </CardContent>
    </Card>
  );
}

function ModelReport({ id, metrics }: { id: string; metrics: ModelMetrics }) {
  const { test, training, curves } = metrics;
  const diff = test.difference_lightgbm_minus_lr;
  const v = verdict(diff.roc_auc_ci95);
  const at = test.at_threshold_lightgbm;
  const nChurners = Math.round(test.n * test.positive_rate);
  const importance = metrics.global_importance_top10;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <ScoreCard
          title="ROC-AUC"
          lgbm={test.lightgbm.roc_auc}
          lr={test.logistic_regression.roc_auc}
          lgbmCi={test.lightgbm.roc_auc_ci95}
          lrCi={test.logistic_regression.roc_auc_ci95}
          note="Chance that a churner gets a higher score than a non-churner (0.5 = random)."
        />
        <ScoreCard
          title="PR-AUC"
          lgbm={test.lightgbm.pr_auc}
          lr={test.logistic_regression.pr_auc}
          lgbmCi={test.lightgbm.pr_auc_ci95}
          lrCi={test.logistic_regression.pr_auc_ci95}
          note={`Average precision; a random ranking scores the churn rate (${pct(test.positive_rate, 1)}).`}
        />
        <Card size="sm">
          <CardHeader>
            <CardDescription>LightGBM minus baseline (ROC-AUC)</CardDescription>
            <CardTitle className="text-2xl tabular-nums">{signed(diff.roc_auc)}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            <p className="tabular-nums text-muted-foreground">95% CI {ci(diff.roc_auc_ci95)} (paired bootstrap)</p>
            <p className={`font-medium ${v.tone}`}>{v.text}.</p>
            <p className="text-xs text-muted-foreground">
              PR-AUC difference {signed(diff.pr_auc)} {ci(diff.pr_auc_ci95)}.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Who gets flagged</CardTitle>
          <CardDescription>
            Threshold {num(at.threshold, 2)}, chosen on the training split (out-of-fold predictions, best F1).
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm leading-relaxed">
          Flags <strong>{at.n_flagged.toLocaleString()}</strong> of {test.n.toLocaleString()} test customers.{" "}
          <strong>{pct(at.precision)}</strong> of flagged customers actually churned, and the model catches{" "}
          <strong>{pct(at.recall)}</strong> of the {nChurners.toLocaleString()} churners.
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>ROC curve</CardTitle>
            <CardDescription>True-positive rate against false-positive rate on the test split.</CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart responsive style={{ width: "100%", height: 280 }} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickCount={6}
                  label={{ value: "False-positive rate", position: "insideBottom", offset: -12 }} />
                <YAxis type="number" domain={[0, 1]} tickCount={6} />
                <Tooltip formatter={(value) => num(Number(value), 3)} labelFormatter={(l) => `FPR ${num(Number(l), 3)}`} />
                <Legend verticalAlign="top" height={28} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#bbb" strokeDasharray="4 4" />
                <Line data={curves.lightgbm.roc} dataKey="tpr" name="LightGBM" stroke={MODEL} dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line data={curves.logistic_regression.roc} dataKey="tpr" name="Baseline (LR)" stroke={BASELINE} dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Precision–recall curve</CardTitle>
            <CardDescription>Dashed line: the churn rate, what a random ranking would get.</CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart responsive style={{ width: "100%", height: 280 }} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="recall" domain={[0, 1]} tickCount={6}
                  label={{ value: "Recall", position: "insideBottom", offset: -12 }} />
                <YAxis type="number" domain={[0, 1]} tickCount={6} />
                <Tooltip formatter={(value) => num(Number(value), 3)} labelFormatter={(l) => `Recall ${num(Number(l), 3)}`} />
                <Legend verticalAlign="top" height={28} />
                <ReferenceLine y={test.positive_rate} stroke="#bbb" strokeDasharray="4 4" />
                <Line data={curves.lightgbm.pr} dataKey="precision" name="LightGBM" stroke={MODEL} dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line data={curves.logistic_regression.pr} dataKey="precision" name="Baseline (LR)" stroke={BASELINE} dot={false} strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Top 10 factors</CardTitle>
          <CardDescription>
            Average size of each column&apos;s push on the churn score across training customers (LightGBM
            contributions; longer bar = more influence, in either direction).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <BarChart
            responsive
            style={{ width: "100%", height: Math.max(220, importance.length * 32) }}
            data={importance}
            layout="vertical"
            margin={{ top: 0, right: 20, bottom: 0, left: 10 }}
          >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value) => num(Number(value), 3)} />
              <Bar dataKey="mean_abs_contribution" name="Average impact" fill={MODEL} radius={[0, 4, 4, 0]} isAnimationActive={false} />
          </BarChart>
        </CardContent>
      </Card>

      <div className="space-y-1 text-sm text-muted-foreground">
        <p>
          Tuned with Optuna: {training.optuna.n_trials_completed} trials, 5-fold CV ROC-AUC{" "}
          {num(training.lightgbm.cv_auc_mean)} ± {num(training.lightgbm.cv_auc_std)} on the training split (
          {metrics.n_train.toLocaleString()} rows; baseline {num(training.logistic_regression.cv_auc_mean)}). Test
          split: {metrics.n_test.toLocaleString()} rows.
          {training.optuna.stopped_by_timeout ? " Optuna stopped early because of the time limit." : ""}
        </p>
        {metrics.notes.map((n) => (
          <p key={n}>Note: {n}</p>
        ))}
      </div>

      <Link href={`/datasets/${id}/predict`} className={buttonVariants()}>
        Score new customers
        <ArrowRight aria-hidden />
      </Link>
    </div>
  );
}
