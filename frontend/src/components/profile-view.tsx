"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { HelpCircle, Play } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ErrorAlert, Loading } from "@/components/status";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import { FLAG_LABEL, FLAG_ORDER, LEAKAGE_FLAGS, flagExplanation } from "@/lib/flags";
import { num, pct } from "@/lib/format";
import { datasetKey, useDataset, useProfile } from "@/lib/queries";
import type { ColumnFlag, Dataset, Profile } from "@/lib/types";
import { cn } from "@/lib/utils";

const selectClass =
  "h-9 w-full rounded-lg border border-input bg-background px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

export function ProfileView({ id }: { id: string }) {
  const dataset = useDataset(id);
  if (dataset.isPending) return <Loading label="Loading dataset…" />;
  if (dataset.isError) return <ErrorAlert error={dataset.error} title="Could not load the dataset" />;
  return <ProfileForm dataset={dataset.data} />;
}

function ProfileForm({ dataset }: { dataset: Dataset }) {
  const candidates = dataset.target_candidates;
  const initialTarget = dataset.target ?? dataset.suggested_target ?? candidates[0]?.name ?? null;
  const [target, setTarget] = useState<string | null>(initialTarget);
  const candidate = candidates.find((c) => c.name === target);
  const [positive, setPositive] = useState<string | null>(
    dataset.target === initialTarget && dataset.positive_label ? dataset.positive_label : (candidate?.suggested_positive ?? null),
  );
  const profile = useProfile(dataset.id, target, positive);

  if (candidates.length === 0) {
    return (
      <ErrorAlert
        title="No possible churn column"
        error="No column has exactly two values. The file needs a column that says whether each customer churned."
      />
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>What should the model predict?</CardTitle>
          <CardDescription>Columns with exactly two values can be the churn column.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <label className="space-y-1.5 text-sm">
            <span className="font-medium">Churn column (target)</span>
            <select
              className={selectClass}
              value={target ?? ""}
              onChange={(e) => {
                const next = candidates.find((c) => c.name === e.target.value);
                setTarget(e.target.value);
                setPositive(next?.suggested_positive ?? null);
              }}
            >
              {candidates.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                  {c.name_suggests_target ? " (suggested)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1.5 text-sm">
            <span className="font-medium">Which value means the customer churned?</span>
            <select
              className={selectClass}
              value={positive ?? ""}
              onChange={(e) => setPositive(e.target.value || null)}
            >
              <option value="">Choose a value…</option>
              {candidate?.values.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </CardContent>
      </Card>

      {!positive && <p className="text-sm text-muted-foreground">Choose which value means churn to profile the columns.</p>}
      {profile.isFetching && <Loading label="Checking every column on the training split… (10–60 s)" />}
      <ErrorAlert error={profile.error} title="Cannot use this target" />
      {profile.data?.profile && !profile.isFetching && (
        <ColumnReview
          key={`${target}|${positive}`}
          dataset={profile.data}
          profile={profile.data.profile}
          target={target!}
          positive={positive}
        />
      )}
    </div>
  );
}

function FlagBadge({ flag }: { flag: ColumnFlag }) {
  return (
    <Badge
      variant={LEAKAGE_FLAGS.has(flag) ? "destructive" : flag === "check_timing" ? "outline" : "secondary"}
      className={cn(flag === "check_timing" && "border-amber-300 text-amber-800")}
    >
      {FLAG_LABEL[flag]}
    </Badge>
  );
}

function ColumnReview({
  dataset,
  profile,
  target,
  positive,
}: {
  dataset: Dataset;
  profile: Profile;
  target: string;
  positive: string | null;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [keep, setKeep] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(profile.columns.map((c) => [c.name, c.suggested === "use"])),
  );
  const features = profile.columns.filter((c) => keep[c.name]).map((c) => c.name);
  const flagsPresent = FLAG_ORDER.filter((f) => profile.columns.some((c) => c.flags.includes(f)));
  const isTraining = dataset.status === "training";

  const train = useMutation({
    mutationFn: () => api.train(dataset.id, { target, positive_label: positive, features }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetKey(dataset.id) });
      router.push(`/datasets/${dataset.id}/model`);
    },
  });

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">
        {profile.n_rows_raw.toLocaleString()} rows, {profile.n_duplicates_removed.toLocaleString()} exact duplicates
        removed. Training split {profile.n_train.toLocaleString()} rows (churn rate {pct(profile.positive_rate_train, 1)}
        ), test split {profile.n_test.toLocaleString()} rows. All checks below use the training split only.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Columns</CardTitle>
          <CardDescription>
            Flagged columns are unticked by default. &quot;AUC alone&quot; is how well a model that sees only this
            column separates churners (0.5 = no signal, 1.0 = perfect).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">Keep</TableHead>
                  <TableHead>Column</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead className="text-right">Missing</TableHead>
                  <TableHead className="text-right">AUC alone</TableHead>
                  <TableHead>Flags</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {profile.columns.map((c) => (
                  <TableRow key={c.name} className={cn(!keep[c.name] && "text-muted-foreground")}>
                    <TableCell>
                      <input
                        type="checkbox"
                        aria-label={`Keep ${c.name}`}
                        className="size-4 accent-primary"
                        checked={keep[c.name]}
                        onChange={(e) => setKeep((k) => ({ ...k, [c.name]: e.target.checked }))}
                      />
                    </TableCell>
                    <TableCell className="font-medium whitespace-normal">
                      {c.name}
                      {c.flags.includes("check_timing") && (
                        <span className="mt-0.5 flex items-center gap-1 text-xs font-normal text-amber-800">
                          <HelpCircle className="size-3" aria-hidden />
                          Is this known before the prediction date?
                        </span>
                      )}
                    </TableCell>
                    <TableCell>{c.role}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(c.missing_rate, 1)}</TableCell>
                    <TableCell className="text-right tabular-nums">{num(c.single_column_auc, 2)}</TableCell>
                    <TableCell className="whitespace-normal">
                      <div className="flex flex-wrap gap-1">
                        {c.flags.map((f) => (
                          <FlagBadge key={f} flag={f} />
                        ))}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          {flagsPresent.length > 0 && (
            <ul className="space-y-1.5 text-sm">
              {flagsPresent.map((f) => (
                <li key={f} className="flex flex-wrap items-start gap-2">
                  <FlagBadge flag={f} />
                  <span className="text-muted-foreground">{flagExplanation(f, profile.thresholds)}</span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {isTraining && (
        <Alert>
          <AlertTitle>A model is training</AlertTitle>
          <AlertDescription>
            <Link href={`/datasets/${dataset.id}/model`} className="underline">
              Follow its progress
            </Link>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => train.mutate()} disabled={features.length === 0 || train.isPending || isTraining}>
          <Play aria-hidden />
          Train on {features.length} column{features.length === 1 ? "" : "s"}
        </Button>
        {dataset.model && (
          <Link href={`/datasets/${dataset.id}/model`} className={buttonVariants({ variant: "outline" })}>
            View current model
          </Link>
        )}
        <span className="text-sm text-muted-foreground">
          Training tunes LightGBM with Optuna and usually takes a few minutes.
          {dataset.model ? " Training again replaces the current model." : ""}
        </span>
      </div>
      <ErrorAlert error={train.error} title="Could not start training" />
    </div>
  );
}
