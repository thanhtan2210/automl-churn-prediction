"use client";

import { FileSpreadsheet } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { useDataset } from "@/lib/queries";
import { cn } from "@/lib/utils";

const STATUS_LABEL = {
  uploaded: "uploaded",
  profiled: "profiled",
  training: "training…",
  trained: "trained",
  failed: "failed",
} as const;

export function DatasetNav({ id }: { id: string }) {
  const pathname = usePathname();
  const { data } = useDataset(id);
  const steps = [
    { href: `/datasets/${id}`, label: "1. Profile" },
    { href: `/datasets/${id}/model`, label: "2. Model" },
    { href: `/datasets/${id}/predict`, label: "3. Predict" },
  ];

  return (
    <div className="mb-6 flex flex-col gap-3 border-b pb-4 md:flex-row md:items-center md:justify-between">
      <div className="flex min-w-0 items-center gap-2">
        <FileSpreadsheet className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <span className="truncate font-medium">{data?.filename ?? "Dataset"}</span>
        {data && (
          <>
            <span className="text-sm text-muted-foreground">{data.n_rows.toLocaleString()} rows</span>
            <Badge variant={data.status === "failed" ? "destructive" : "secondary"}>{STATUS_LABEL[data.status]}</Badge>
          </>
        )}
      </div>
      <nav className="flex gap-1 text-sm" aria-label="Steps">
        {steps.map((s) => (
          <Link
            key={s.href}
            href={s.href}
            className={cn(
              "rounded-md px-3 py-1.5 transition-colors",
              pathname === s.href ? "bg-primary text-primary-foreground" : "hover:bg-muted",
            )}
          >
            {s.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
