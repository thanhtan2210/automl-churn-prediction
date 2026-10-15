"use client";

import { FileSpreadsheet, FileUp } from "lucide-react";
import { useRef, useState } from "react";

import { cn } from "@/lib/utils";

const ACCEPT = ".csv,.xlsx";

function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function FileDrop({
  onFile,
  disabled,
  hint,
  compact = false,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
  hint?: React.ReactNode;
  compact?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [picked, setPicked] = useState<File | null>(null);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file && !disabled) {
      setPicked(file);
      onFile(file);
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) input.current?.click();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        pick(e.dataTransfer.files);
      }}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed bg-background px-6 text-center transition-all",
        compact ? "py-6" : "py-12",
        dragging ? "scale-[1.01] border-primary bg-brand-soft" : "border-border hover:border-primary/50 hover:bg-brand-soft/50",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <span className="flex size-12 items-center justify-center rounded-full bg-brand-soft text-primary">
        <FileUp className="size-6" aria-hidden />
      </span>
      <p className="font-medium">
        {dragging ? "Drop the file to upload it" : "Drag a CSV or Excel file here, or click to choose one"}
      </p>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
      {picked && (
        <p className="mt-1 flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-sm">
          <FileSpreadsheet className="size-4 text-muted-foreground" aria-hidden />
          <span className="max-w-60 truncate font-medium">{picked.name}</span>
          <span className="text-muted-foreground">{fileSize(picked.size)}</span>
        </p>
      )}
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
