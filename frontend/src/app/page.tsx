"use client";

import { useEffect, useState } from "react";
import { UploadDropzone } from "@/components/UploadDropzone";
import { VerdictCard } from "@/components/VerdictCard";
import { ApiError, getHealth, verify } from "@/lib/api";
import type { Health, Verdict } from "@/lib/types";

export default function Home() {
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>();
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    getHealth().then(setHealth);
  }, []);

  async function handleFile(file: File) {
    setError(undefined);
    setVerdict(null);
    setBusy(true);
    setProgress(0);
    const isImage = file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".gif");
    if (isImage) setPreviewUrl(URL.createObjectURL(file));
    else setPreviewUrl(undefined);

    try {
      const result = await verify(file, { explain: isImage, onProgress: setProgress });
      setVerdict(result);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "something went wrong";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-2xl px-5 py-12">
      <header className="mb-8 text-center">
        <h1 className="bg-gradient-to-r from-sky-300 to-indigo-300 bg-clip-text text-4xl font-bold text-transparent">
          Veritas
        </h1>
        <p className="mt-2 text-slate-400">
          Deepfake &amp; AI-content detector — upload media for an authenticity verdict, confidence
          breakdown, and a Grad-CAM heatmap.
        </p>
      </header>

      {health && (
        <div className="mb-6 flex flex-wrap justify-center gap-2 text-xs">
          <Pill ok={health.image_model} label="image model" />
          <Pill ok={health.audio_model} label="audio model" />
          <Pill ok={health.explainer} label="heatmaps" />
          <span className="rounded-full bg-white/5 px-2.5 py-1 text-slate-400">cache: {health.cache}</span>
        </div>
      )}
      {health === null && (
        <p className="mb-6 text-center text-sm text-amber-300/80">
          API unreachable — start it with <code className="font-mono">docker compose up</code>.
        </p>
      )}

      <UploadDropzone onFile={handleFile} busy={busy} progress={progress} />

      {error && (
        <div className="mt-6 rounded-xl border border-fake/30 bg-fake/10 px-4 py-3 text-sm text-fake">
          {error}
        </div>
      )}

      {verdict && (
        <div className="mt-8">
          <VerdictCard verdict={verdict} previewUrl={previewUrl} />
        </div>
      )}

      <footer className="mt-12 text-center text-xs text-slate-600">
        Models served via ONNX Runtime · results cached by content hash
      </footer>
    </main>
  );
}

function Pill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`rounded-full px-2.5 py-1 ${ok ? "bg-real/15 text-real" : "bg-white/5 text-slate-500"}`}
    >
      {ok ? "●" : "○"} {label}
    </span>
  );
}
