import type { Verdict } from "@/lib/types";
import { ConfidenceBar } from "./ConfidenceBar";
import { HeatmapView } from "./HeatmapView";

interface Props {
  verdict: Verdict;
  previewUrl?: string;
}

export function VerdictCard({ verdict, previewUrl }: Props) {
  const isFake = verdict.verdict === "fake";
  return (
    <div className="space-y-5 rounded-2xl border border-white/10 bg-panel/70 p-6 shadow-xl">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">Verdict</p>
          <p className={`text-3xl font-bold ${isFake ? "text-fake" : "text-real"}`}>
            {isFake ? "Likely manipulated" : "Likely authentic"}
          </p>
        </div>
        <span
          className={`rounded-full px-3 py-1 text-sm font-semibold ${
            isFake ? "bg-fake/15 text-fake" : "bg-real/15 text-real"
          }`}
        >
          {verdict.verdict.toUpperCase()}
        </span>
      </div>

      <div className="space-y-3">
        <ConfidenceBar label="Confidence" value={verdict.confidence} tone={isFake ? "fake" : "real"} />
        <ConfidenceBar label="Fake probability" value={verdict.fake_probability} tone="neutral" />
      </div>

      {verdict.modality === "image" && previewUrl && (
        <HeatmapView original={previewUrl} overlay={verdict.overlay} />
      )}

      {verdict.modality === "video" && verdict.frames && (
        <div>
          <p className="mb-2 text-sm text-slate-300">
            {verdict.frames_analyzed} frames analysed — per-frame fake probability
          </p>
          <div className="flex h-16 items-end gap-1">
            {verdict.frames.map((f) => (
              <div
                key={f.index}
                title={`frame ${f.index}: ${(f.fake_probability * 100).toFixed(0)}%`}
                className="flex-1 rounded-t bg-gradient-to-t from-sky-500/40 to-fake/80"
                style={{ height: `${Math.max(4, f.fake_probability * 100)}%` }}
              />
            ))}
          </div>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-3 text-sm text-slate-300 sm:grid-cols-4">
        <Meta label="Modality" value={verdict.modality} />
        <Meta label="Model" value={verdict.model} />
        <Meta label="Latency" value={`${verdict.latency_ms.toFixed(0)} ms`} />
        <Meta label="Cache" value={verdict.cached ? "hit" : "miss"} />
      </dl>
      <p className="truncate font-mono text-[11px] text-slate-500" title={verdict.content_sha256}>
        sha256: {verdict.content_sha256}
      </p>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-black/20 px-3 py-2">
      <dt className="text-[11px] uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="truncate font-medium text-slate-200">{value}</dd>
    </div>
  );
}
