"use client";

import { FileUp } from "lucide-react";
import { useRef, useState } from "react";

import { cn } from "@/lib/utils";

const ACCEPT = ".csv,.xlsx";

export function FileDrop({
  onFile,
  disabled,
  hint,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
  hint?: React.ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file && !disabled) onFile(file);
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
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed bg-background px-6 py-10 text-center transition-colors",
        dragging ? "border-primary bg-muted" : "border-border hover:bg-muted/50",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <FileUp className="size-8 text-muted-foreground" aria-hidden />
      <p className="font-medium">Drop a CSV or XLSX file here, or click to choose one</p>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
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
