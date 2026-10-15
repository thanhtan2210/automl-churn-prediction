"use client";

import { Check } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Dataset } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Step {
  label: string;
  href: string | null;
  done: boolean;
  disabledReason: string | null;
}

function steps(dataset: Dataset | undefined, id: string | undefined): Step[] {
  const base = id ? `/datasets/${id}` : null;
  const hasModel = Boolean(dataset?.model);
  const modelReachable = hasModel || dataset?.status === "training" || dataset?.status === "failed";
  return [
    { label: "Upload", href: "/", done: Boolean(id), disabledReason: null },
    {
      label: "Profile",
      href: base,
      done: Boolean(dataset?.profile),
      disabledReason: id ? null : "Upload a file or pick a sample first.",
    },
    {
      label: "Model",
      href: base && `${base}/model`,
      done: hasModel,
      disabledReason: !id ? "Upload a file first." : modelReachable ? null : "Train a model on the Profile step first.",
    },
    {
      label: "Predict",
      href: base && `${base}/predict`,
      done: false,
      disabledReason: !id ? "Upload a file first." : hasModel ? null : "Train a model first, then score new customers.",
    },
  ];
}

export function StepBar({ dataset, id }: { dataset?: Dataset; id?: string }) {
  const pathname = usePathname();
  const items = steps(dataset, id);

  return (
    <nav aria-label="Steps" className="mb-8">
      <ol className="flex items-center gap-2 overflow-x-auto rounded-2xl border bg-background p-2 shadow-xs sm:gap-3">
        {items.map((step, i) => {
          const current = step.href === pathname;
          const disabled = Boolean(step.disabledReason) || !step.href;
          const inner = (
            <span
              className={cn(
                "flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors sm:px-4",
                current && "bg-primary text-primary-foreground shadow-sm",
                !current && !disabled && "hover:bg-brand-soft",
                disabled && "cursor-not-allowed text-muted-foreground opacity-60",
              )}
            >
              <span
                className={cn(
                  "flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                  current && "border-primary-foreground/40 bg-primary-foreground/15",
                  !current && step.done && "border-primary bg-primary text-primary-foreground",
                  !current && !step.done && "bg-background",
                )}
              >
                {step.done && !current ? <Check className="size-4" aria-label="done" /> : i + 1}
              </span>
              <span>{step.label}</span>
            </span>
          );
          return (
            <li key={step.label} className="flex flex-1 items-center gap-2 sm:gap-3">
              {disabled ? (
                <Tooltip>
                  <TooltipTrigger render={<span tabIndex={0} aria-disabled="true" />}>{inner}</TooltipTrigger>
                  <TooltipContent>{step.disabledReason}</TooltipContent>
                </Tooltip>
              ) : (
                <Link href={step.href!} aria-current={current ? "step" : undefined}>
                  {inner}
                </Link>
              )}
              {i < items.length - 1 && <span className="hidden h-px flex-1 bg-border sm:block" aria-hidden />}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
