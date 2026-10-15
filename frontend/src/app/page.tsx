"use client";

import { useMutation } from "@tanstack/react-query";
import { ArrowRight, ExternalLink, ListChecks, ShieldAlert, Smartphone, Sparkles, Users, Wifi } from "lucide-react";
import { useRouter } from "next/navigation";

import { FileDrop } from "@/components/file-drop";
import { HowToRead } from "@/components/how-to-read";
import { ErrorAlert, Loading } from "@/components/status";
import { StepBar } from "@/components/step-bar";
import { api } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

const RESULTS_URL = "https://github.com/thanhtan2210/automl-churn-prediction#results";

const SAMPLES = [
  {
    key: "telco" as const,
    title: "IBM Telco customers",
    icon: Wifi,
    description: "Fictional customers of a California phone and internet provider: contract, services and charges.",
    source: "IBM Cognos Analytics sample data",
    license: "No published license (used for learning)",
  },
  {
    key: "iranian" as const,
    title: "Iranian telecom subscribers",
    icon: Smartphone,
    description: "Call, SMS and complaint activity of mobile subscribers, with who left a few months later.",
    source: "UCI Machine Learning Repository #563",
    license: "CC BY 4.0",
  },
];

const FLOW = [
  { icon: ListChecks, title: "Check columns", text: "Spot columns that give the answer away" },
  { icon: Sparkles, title: "Train", text: "Learn who left from past customers" },
  { icon: Users, title: "Score customers", text: "Rank new customers with reasons" },
];

const FAQ = [
  {
    q: "What file can I upload?",
    a: "A CSV or Excel file with one row per customer and one column that says whether the customer left (for example Yes/No or 1/0). The first sheet of an Excel file is used.",
  },
  {
    q: "What happens after I upload?",
    a: "You pick the column that says who left. The app then checks every other column, trains a model on most of the rows, and tests it on rows it never saw.",
  },
  {
    q: "Where does my data go?",
    a: "Only to the API running on this machine. It stores the file and the trained model in its local data folder.",
  },
];

export default function UploadPage() {
  const router = useRouter();
  const upload = useMutation({
    mutationFn: (input: File | "telco" | "iranian") =>
      typeof input === "string" ? api.uploadSample(input) : api.uploadFile(input),
    onSuccess: (data: UploadResponse) => router.push(`/datasets/${data.id}`),
  });

  return (
    <div className="space-y-8">
      <StepBar />

      <section className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Who is about to cancel — and why?</h1>
          <p className="max-w-2xl text-lg text-muted-foreground">
            Upload your subscriber list. The app checks the columns, learns from customers who already left, and
            ranks the others by how likely they are to leave.
          </p>
        </div>
        <HowToRead items={FAQ} />
      </section>

      <ol className="grid gap-3 sm:grid-cols-3">
        {FLOW.map((step, i) => (
          <li key={step.title} className="flex items-center gap-3 rounded-2xl border bg-card p-4 shadow-xs">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-soft text-primary">
              <step.icon className="size-5" aria-hidden />
            </span>
            <div className="min-w-0">
              <p className="text-sm font-semibold">
                {i + 1}. {step.title}
              </p>
              <p className="text-sm text-muted-foreground">{step.text}</p>
            </div>
            {i < FLOW.length - 1 && <ArrowRight className="ml-auto hidden size-4 text-muted-foreground sm:block" aria-hidden />}
          </li>
        ))}
      </ol>

      <section className="space-y-4">
        <FileDrop
          onFile={(file) => upload.mutate(file)}
          disabled={upload.isPending}
          hint="One row per customer, with a column that says who left."
        />
        {upload.isPending && <Loading label="Uploading and reading the file…" />}
        <ErrorAlert error={upload.error} title="The file could not be uploaded" />
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">No file at hand? Try a sample</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {SAMPLES.map((s) => (
            <button
              key={s.key}
              type="button"
              disabled={upload.isPending}
              onClick={() => upload.mutate(s.key)}
              className="group flex gap-4 rounded-2xl border bg-card p-5 text-left shadow-xs transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-60"
            >
              <span className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-brand-soft text-primary">
                <s.icon className="size-6" aria-hidden />
              </span>
              <span className="min-w-0 space-y-1">
                <span className="flex items-center gap-1.5 font-semibold">
                  {s.title}
                  <ArrowRight className="size-4 text-primary transition-transform group-hover:translate-x-0.5" aria-hidden />
                </span>
                <span className="block text-sm text-muted-foreground">{s.description}</span>
                <span className="block text-xs text-muted-foreground">
                  Source: {s.source} · {s.license}
                </span>
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className="flex gap-4 rounded-2xl border border-warn/25 bg-warn-soft p-5">
        <ShieldAlert className="mt-0.5 size-6 shrink-0 text-warn" aria-hidden />
        <div className="space-y-1.5 text-sm">
          <p className="font-semibold">What makes this different</p>
          <p>
            Many churn files contain columns that are only filled in after a customer has left, such as a churn
            reason or a score from another model. A model that sees them looks perfect in testing and fails on real
            customers. This app finds those columns and leaves them out before training.
          </p>
          <a href={RESULTS_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-medium text-primary hover:underline">
            See what that changes on the sample data
            <ExternalLink className="size-3.5" aria-hidden />
          </a>
        </div>
      </section>
    </div>
  );
}
