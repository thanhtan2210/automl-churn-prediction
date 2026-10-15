"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CircleHelp, Clock, Columns3, EyeOff, Play, Target } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { HowToRead } from "@/components/how-to-read";
import { EmptyState, ErrorAlert, PageSkeleton } from "@/components/status";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import { FLAG_LABEL, LEAKAGE_FLAGS, ROLE_LABEL, flagExplanation, isLeak, leakSentence, otherReason } from "@/lib/flags";
import { duration, pct } from "@/lib/format";
import { datasetKey, useDataset, useProfile } from "@/lib/queries";
import type { ColumnFlag, ColumnProfile, Dataset, Profile } from "@/lib/types";
import { cn } from "@/lib/utils";

const selectClass =
  "h-10 w-full rounded-xl border border-input bg-background px-3 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50";

export function ProfileView({ id }: { id: string }) {
  const dataset = useDataset(id);
  if (dataset.isPending) return <PageSkeleton label="Loading the dataset…" />;
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
      <EmptyState icon={Target} title="No column says who left">
        The file needs a column with exactly two values (for example Yes/No or 1/0) that says whether each customer
        left. Add one and upload the file again.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-6 pb-24">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Check the columns</h1>
          <p className="text-muted-foreground">
            Tell the app which column says who left. It then checks every other column before training.
          </p>
        </div>
        {profile.data?.profile && <ProfileFaq profile={profile.data.profile} />}
      </div>

      <div className="grid gap-4 rounded-2xl border bg-card p-5 shadow-xs md:grid-cols-2">
        <label className="space-y-1.5 text-sm">
          <span className="font-medium">Which column says who left?</span>
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
          <span className="font-medium">Which value means the customer left?</span>
          <select className={selectClass} value={positive ?? ""} onChange={(e) => setPositive(e.target.value || null)}>
            <option value="">Choose a value…</option>
            {candidate?.values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      </div>

      {!positive && <p className="text-sm text-muted-foreground">Choose which value means the customer left.</p>}
      {profile.isFetching && <PageSkeleton label="Checking every column on the training rows… (10–60 s)" blocks={2} />}
      <ErrorAlert error={profile.error} title="This column cannot be used as the answer" />
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

function ProfileFaq({ profile }: { profile: Profile }) {
  const leaks = profile.columns.filter(isLeak).length;
  const other = Object.keys(profile.suggested_excluded).length - leaks;
  const t = profile.thresholds;
  return (
    <HowToRead
      intro="The checks below use only the rows the model learns from; the rows kept aside for testing are never looked at."
      items={[
        {
          q: "Why were these columns removed?",
          a: `${leaks} column(s) were removed because they give the answer away: they are filled in or calculated after a customer has left, so a model would look perfect in testing and fail on real customers. ${other} more were removed because they are IDs, locations or have one value.`,
        },
        {
          q: "What does 'predictive power alone' mean?",
          a: `It is how well a model that sees only this one column can tell leavers from stayers, from 0.5 (no help) to 1.0 (perfect). Ordinary columns score well below the point where the app gets suspicious: ${t.likely_leakage_auc}.`,
        },
        {
          q: "Should I keep a column the app flagged?",
          a: "Only if you are sure its value is known before you would contact the customer. If it is written down when or after the customer leaves, keep it out — the results would look better than they really are.",
        },
        {
          q: "What does the timing question mean?",
          a: "Some column names suggest an outcome or a score. The app cannot know when such a column was recorded, so it asks you. Untick it if the value is only known after the prediction date.",
        },
      ]}
    />
  );
}

function FlagBadge({ flag, profile }: { flag: ColumnFlag; profile: Profile }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Badge
            variant="outline"
            className={cn(
              "cursor-help",
              LEAKAGE_FLAGS.has(flag) && "border-warn/40 bg-warn-soft text-warn",
              flag === "check_timing" && "border-amber-300 bg-amber-50 text-amber-800",
            )}
          />
        }
      >
        {FLAG_LABEL[flag]}
      </TooltipTrigger>
      <TooltipContent>{flagExplanation(flag, profile.thresholds)}</TooltipContent>
    </Tooltip>
  );
}

function PowerBar({ auc, profile }: { auc: number | null; profile: Profile }) {
  if (auc === null) return <span className="text-muted-foreground">—</span>;
  const width = Math.max(0, Math.min(1, (auc - 0.5) / 0.5)) * 100;
  const suspicious = auc >= profile.thresholds.likely_leakage_auc;
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 overflow-hidden rounded-full bg-muted" aria-hidden>
        <div className={cn("h-full rounded-full", suspicious ? "bg-warn" : "bg-primary")} style={{ width: `${width}%` }} />
      </div>
      <span className="w-9 text-right text-xs tabular-nums">{auc.toFixed(2)}</span>
    </div>
  );
}

function ColumnTable({
  columns,
  keep,
  setKeep,
  profile,
}: {
  columns: ColumnProfile[];
  keep: Record<string, boolean>;
  setKeep: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  profile: Profile;
}) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-14">Keep</TableHead>
            <TableHead>Column</TableHead>
            <TableHead>Type</TableHead>
            <TableHead className="text-right">Empty</TableHead>
            <TableHead>
              <Tooltip>
                <TooltipTrigger render={<span className="inline-flex cursor-help items-center gap-1" />}>
                  Predictive power alone
                  <CircleHelp className="size-3.5" aria-hidden />
                </TooltipTrigger>
                <TooltipContent>
                  ROC-AUC of a small model that sees only this column, on the training rows: 0.5 = no help, 1.0 =
                  tells who left perfectly.
                </TooltipContent>
              </Tooltip>
            </TableHead>
            <TableHead>Notes</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {columns.map((c) => (
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
                  <span className="mt-1 flex w-fit items-center gap-1 rounded-md bg-amber-50 px-1.5 py-0.5 text-xs font-medium text-amber-800">
                    <Clock className="size-3" aria-hidden />
                    Is this known before the prediction date?
                  </span>
                )}
              </TableCell>
              <TableCell>{ROLE_LABEL[c.role]}</TableCell>
              <TableCell className="text-right tabular-nums">{pct(c.missing_rate, 1)}</TableCell>
              <TableCell>
                <PowerBar auc={c.single_column_auc} profile={profile} />
              </TableCell>
              <TableCell className="whitespace-normal">
                <div className="flex flex-wrap gap-1">
                  {c.flags.map((f) => (
                    <FlagBadge key={f} flag={f} profile={profile} />
                  ))}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
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
  const used = profile.columns.filter((c) => c.suggested === "use");
  const leaks = profile.columns.filter(isLeak);
  const other = profile.columns.filter((c) => c.suggested === "exclude" && !isLeak(c));
  const isTraining = dataset.status === "training";
  const lastSeconds = dataset.model?.metrics.timing_seconds.total;

  const train = useMutation({
    mutationFn: () => api.train(dataset.id, { target, positive_label: positive, features }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetKey(dataset.id) });
      router.push(`/datasets/${dataset.id}/model`);
    },
  });

  const groups = [
    { key: "leak", title: `Removed: gives the answer away (${leaks.length})`, columns: leaks },
    { key: "used", title: `Used for training (${used.length})`, columns: used },
    { key: "other", title: `Removed: ID, location or same value (${other.length})`, columns: other },
  ].filter((g) => g.columns.length > 0);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-3 rounded-2xl border border-warn/30 bg-warn-soft p-5 lg:col-span-2">
          <p className="flex items-center gap-2 font-semibold text-warn">
            <AlertTriangle className="size-5" aria-hidden />
            {leaks.length === 0
              ? "No column gives the answer away"
              : `${leaks.length} column${leaks.length === 1 ? "" : "s"} removed because they give the answer away`}
          </p>
          {leaks.length > 0 ? (
            <ul className="space-y-2 text-sm">
              {leaks.map((c) => (
                <li key={c.name}>
                  <code className="rounded bg-background/70 px-1.5 py-0.5 font-semibold">{c.name}</code>{" "}
                  <span className="text-foreground/80">{leakSentence(c)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-foreground/80">
              No single column predicts who left suspiciously well. Still check columns marked with a timing question.
            </p>
          )}
        </div>
        <div className="grid gap-4">
          <div className="flex items-center gap-3 rounded-2xl border bg-card p-4 shadow-xs">
            <EyeOff className="size-5 shrink-0 text-muted-foreground" aria-hidden />
            <p className="text-sm">
              <span className="text-2xl font-semibold tabular-nums">{other.length}</span> column
              {other.length === 1 ? "" : "s"} removed as IDs, locations or constants
            </p>
          </div>
          <div className="flex items-center gap-3 rounded-2xl border bg-card p-4 shadow-xs">
            <Columns3 className="size-5 shrink-0 text-primary" aria-hidden />
            <p className="text-sm">
              <span className="text-2xl font-semibold tabular-nums">{used.length}</span> column
              {used.length === 1 ? "" : "s"} will be used
            </p>
          </div>
        </div>
      </div>

      <p className="text-sm text-muted-foreground">
        {profile.n_rows_raw.toLocaleString()} rows ({profile.n_duplicates_removed.toLocaleString()} exact duplicates
        removed). The model learns from {profile.n_train.toLocaleString()} customers, of whom{" "}
        {pct(profile.positive_rate_train, 1)} left, and is tested on {profile.n_test.toLocaleString()} it never sees.
      </p>

      <Accordion multiple defaultValue={["leak", "used"]} className="space-y-3">
        {groups.map((g) => (
          <AccordionItem key={g.key} value={g.key} className="rounded-2xl border bg-card px-4 shadow-xs">
            <AccordionTrigger className="text-base">{g.title}</AccordionTrigger>
            <AccordionContent>
              {g.key === "other" && (
                <ul className="mb-3 space-y-1 text-sm text-muted-foreground">
                  {g.columns.map((c) => (
                    <li key={c.name}>
                      <span className="font-medium text-foreground">{c.name}</span>: {otherReason(c)}
                    </li>
                  ))}
                </ul>
              )}
              <ColumnTable columns={g.columns} keep={keep} setKeep={setKeep} profile={profile} />
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>

      <ErrorAlert error={train.error} title="Training could not start" />

      <div className="sticky bottom-4 z-10 flex flex-col gap-3 rounded-2xl border bg-background/95 p-4 shadow-lg backdrop-blur sm:flex-row sm:items-center">
        <Button size="lg" onClick={() => train.mutate()} disabled={features.length === 0 || train.isPending || isTraining}>
          <Play aria-hidden />
          Train on {features.length} column{features.length === 1 ? "" : "s"}
        </Button>
        <span className="hidden text-sm text-muted-foreground sm:inline">
          {isTraining
            ? "A model is training right now."
            : lastSeconds
              ? `The last training on this file took ${duration(lastSeconds)}.`
              : "Training usually takes a few minutes."}
          {dataset.model && !isTraining ? " Training again replaces the current model." : ""}
        </span>
        {(dataset.model || isTraining) && (
          <Link href={`/datasets/${dataset.id}/model`} className={cn(buttonVariants({ variant: "outline" }), "sm:ml-auto")}>
            {isTraining ? "Follow training" : "View current model"}
          </Link>
        )}
      </div>
    </div>
  );
}
