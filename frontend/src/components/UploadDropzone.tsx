"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  progress: number;
}

const ACCEPT = "image/*,audio/*,video/*,.gif";

export function UploadDropzone({ onFile, busy, progress }: Props) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (files && files.length > 0) onFile(files[0]);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (!busy) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !busy && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !busy && inputRef.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 text-center transition ${
        drag ? "border-sky-400 bg-sky-400/10" : "border-white/15 hover:border-white/30 hover:bg-white/5"
      } ${busy ? "pointer-events-none opacity-70" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <UploadIcon />
      <p className="mt-4 text-lg font-medium text-slate-100">
        {busy ? "Analysing…" : "Drop an image, audio clip, or video"}
      </p>
      <p className="mt-1 text-sm text-slate-400">or click to choose a file · PNG · WAV · MP4 · GIF</p>

      {busy && (
        <div className="mt-6 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-sky-400 transition-[width] duration-200"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

function UploadIcon() {
  return (
    <svg
      width="40"
      height="40"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      className="text-sky-300"
      aria-hidden="true"
    >
      <path d="M12 16V4m0 0L7 9m5-5 5 5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
    </svg>
  );
}
