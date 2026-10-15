"use client";

import { useMutation } from "@tanstack/react-query";
import { Database } from "lucide-react";
import { useRouter } from "next/navigation";

import { FileDrop } from "@/components/file-drop";
import { ErrorAlert, Loading } from "@/components/status";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { UploadResponse } from "@/lib/types";

const SAMPLES = [
  {
    key: "telco" as const,
    label: "Try IBM Telco",
    description: "Fictional customers of a California telco (IBM Cognos sample data, no published license).",
  },
  {
    key: "iranian" as const,
    label: "Try UCI Iranian",
    description: "Subscribers of an Iranian telecom, usage over 9 months (UCI ML Repository #563, CC BY 4.0).",
  },
];

const STEPS = [
  ["Profile", "Pick the churn column. Every column is checked on the training split for leakage."],
  ["Train", "LightGBM tuned with Optuna, compared with a logistic-regression baseline on a held-out test split."],
  ["Predict", "Upload new customers to get a churn probability and the top reasons for each one."],
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
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Find out who is likely to cancel — and why</h1>
        <p className="max-w-2xl text-muted-foreground">
          Upload a list of telecom subscribers with a column that says who churned. The app checks the columns for
          leakage, trains a model, and explains each customer&apos;s risk.
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Upload your data</CardTitle>
          <CardDescription>CSV or XLSX, up to 50 MB, at least 200 rows, one row per customer.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <FileDrop
            onFile={(file) => upload.mutate(file)}
            disabled={upload.isPending}
            hint="The first sheet of an Excel file is used."
          />
          <div className="grid gap-3 md:grid-cols-2">
            {SAMPLES.map((s) => (
              <div key={s.key} className="flex flex-col gap-2 rounded-lg border bg-background p-4">
                <Button
                  variant="outline"
                  className="w-fit"
                  disabled={upload.isPending}
                  onClick={() => upload.mutate(s.key)}
                >
                  <Database aria-hidden />
                  {s.label}
                </Button>
                <p className="text-sm text-muted-foreground">{s.description}</p>
              </div>
            ))}
          </div>
          {upload.isPending && <Loading label="Uploading and reading the file…" />}
          <ErrorAlert error={upload.error} title="Upload failed" />
        </CardContent>
      </Card>

      <section className="grid gap-4 md:grid-cols-3">
        {STEPS.map(([title, text], i) => (
          <div key={title} className="rounded-lg border bg-background p-4">
            <p className="text-sm font-medium">
              {i + 1}. {title}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">{text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
