"use client";

import { UploadCloud, XCircle } from "lucide-react";
import { useId, useRef, useState } from "react";
import { APP_COPY, type AppLocale } from "@/lib/app-copy";

export type AcceptedAudioExtension = "mp3" | "wav" | "m4a" | "mp4" | "ogg" | "flac" | "aac" | "webm" | "opus";

export interface UploadDropzoneProps {
  disabled?: boolean;
  disabledReason?: string;
  locale?: AppLocale;
  onDisabledActivate?: () => void;
  onFileSelected?: (file: File) => void;
}

const ACCEPTED_AUDIO_EXTENSIONS: AcceptedAudioExtension[] = ["mp3", "wav", "m4a", "mp4", "ogg", "flac", "aac", "webm", "opus"];
const ACCEPT_ATTRIBUTE =
  ".mp3,.wav,.m4a,.mp4,.ogg,.flac,.aac,.webm,.opus,audio/mpeg,audio/wav,audio/x-wav,audio/mp4,audio/x-m4a,audio/ogg,audio/flac,audio/aac,audio/webm,audio/opus";

export function isAcceptedAudioFile(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return (
    extension !== undefined &&
    ACCEPTED_AUDIO_EXTENSIONS.includes(extension as AcceptedAudioExtension)
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadDropzone({
  disabled = false,
  disabledReason,
  locale = "en",
  onDisabledActivate,
  onFileSelected,
}: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const disabledReasonId = useId();
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const copy = APP_COPY[locale].dropzone;

  const handleFile = (file: File | null) => {
    if (!file) return;

    if (!isAcceptedAudioFile(file)) {
      setSelectedFile(null);
      setError(copy.unsupportedType);
      return;
    }

    setError(null);
    setSelectedFile(file);
    onFileSelected?.(file);
  };

  const openFilePicker = () => {
    if (disabled) {
      onDisabledActivate?.();
      return;
    }
    inputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        disabled={disabled}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          handleFile(file);
        }}
      />

      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        aria-label={copy.uploadLabel}
        aria-describedby={disabled && disabledReason ? disabledReasonId : undefined}
        onClick={openFilePicker}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openFilePicker();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (disabled) return;
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          const nextTarget = event.relatedTarget;
          if (nextTarget && event.currentTarget.contains(nextTarget as Node)) {
            return;
          }
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (disabled) {
            onDisabledActivate?.();
            return;
          }
          const file = event.dataTransfer.files?.[0] ?? null;
          handleFile(file);
        }}
        className={[
          "group relative flex min-h-40 w-full cursor-pointer flex-col items-center justify-center gap-2.5 rounded-lg border border-dashed p-4 text-center outline-none transition-all duration-200 ease-out sm:min-h-44 sm:p-5",
          "hover:border-cyan-300/35 hover:bg-white/[0.035] focus-visible:ring-2 focus-visible:ring-cyan-300/50",
          disabled ? "cursor-not-allowed border-neutral-800 bg-neutral-900/35 opacity-75" : "",
          isDragging
            ? "border-cyan-300/70 bg-cyan-300/[0.08] shadow-[inset_0_0_0_1px_rgba(103,232,249,0.12)]"
            : "border-neutral-700/80 bg-[#151816]/85",
          !isDragging && error ? "border-red-500/70 bg-red-500/10" : "",
        ].join(" ")}
      >
        <div
          className={[
            "rounded-md border p-2 transition-colors",
            isDragging
              ? "border-cyan-300/70 bg-cyan-300/15 text-cyan-200"
              : "border-neutral-700/80 bg-neutral-800/70 text-neutral-300 group-hover:border-neutral-500",
          ].join(" ")}
        >
          {error ? <XCircle className="size-5" /> : <UploadCloud className="size-5" />}
        </div>

        <div className="space-y-1.5">
          <p className="text-sm font-semibold text-neutral-100">
            {isDragging ? copy.dropHere : copy.uploadPrompt}
          </p>
          <p className="text-xs text-neutral-400">
            {copy.or} <span className="text-cyan-300">{copy.browse}</span>
          </p>
        </div>

        <p className="max-w-xl text-[11px] leading-4 text-neutral-500">{copy.acceptedFormats}</p>
        {disabled && disabledReason ? (
          <p id={disabledReasonId} className="text-[11px] font-medium text-amber-200/70">
            {disabledReason}
          </p>
        ) : null}
      </div>

      {error ? (
        <div role="alert" className="mt-3 flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          <XCircle className="size-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      {selectedFile ? (
        <div role="status" aria-label={copy.selectedFile(selectedFile.name)} className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/10 bg-[#171a18] px-3.5 py-2.5 sm:gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor" className="shrink-0 text-neutral-400"><path d="M560-360v-240l80 80 56-56-160-160-160 160 56 56 80-80v240h48Zm-80 200q-83 0-141.5-58.5T280-360v-400h400v400q0 83-58.5 141.5T480-160Zm0-80q50 0 85-35t35-85v-320H360v320q0 50 35 85t85 35ZM200-80q-33 0-56.5-23.5T120-160v-520h80v520h520v80H200Zm280-440Z" /></svg>
            <span className="truncate text-sm text-neutral-200">{selectedFile.name}</span>
          </div>
          <span className="shrink-0 text-xs tabular-nums text-neutral-500 sm:ml-auto">{formatFileSize(selectedFile.size)}</span>
        </div>
      ) : null}
    </div>
  );
}
