"use client";

import { useState } from "react";

interface Props {
  original: string; // object URL of the uploaded image
  overlay?: string | null; // data URL of the Grad-CAM overlay
}

/** Toggles between the original image and the Grad-CAM overlay. */
export function HeatmapView({ original, overlay }: Props) {
  const [showOverlay, setShowOverlay] = useState(true);
  const src = showOverlay && overlay ? overlay : original;

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl border border-white/10 bg-black/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt="analysed media" className="mx-auto max-h-[360px] w-auto object-contain" />
      </div>
      {overlay && (
        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-400">
            Grad-CAM highlights the regions that most drove the verdict.
          </p>
          <button
            onClick={() => setShowOverlay((v) => !v)}
            className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-white/10"
          >
            {showOverlay ? "Show original" : "Show heatmap"}
          </button>
        </div>
      )}
    </div>
  );
}
